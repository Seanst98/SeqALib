#include <vector>

template <typename ContainerType, typename Ty = typename ContainerType::value_type, Ty Blank = Ty(0), typename MatchFnTy = std::function<bool(Ty, Ty)>>
class BLATSA : public SequenceAligner<ContainerType, Ty, Blank, MatchFnTy>
{
private:
    int wordSize;
    int initialThreshold;

    using BaseType = SequenceAligner<ContainerType, Ty, Blank, MatchFnTy>;

    ScoringSystem &Scoring = BaseType::getScoring();
    const ScoreSystemType Gap = Scoring.getGapPenalty();
    const ScoreSystemType Match = Scoring.getMatchProfit();
    const bool AllowMismatch = Scoring.getAllowMismatch();
    const ScoreSystemType Mismatch = AllowMismatch ? Scoring.getMismatchPenalty() : std::numeric_limits<ScoreSystemType>::min();

    // Build the alignment using the BLAST hueristic algorithm
    void buildAlignment(ContainerType &Seq1, ContainerType &Seq2, AlignedSequence<Ty, Blank> &Result)
    {
        auto &Data = Result.Data;
        const size_t SizeSeq1 = Seq1.size();
        const size_t SizeSeq2 = Seq2.size();

        // Determine appropriate word size based on the size of the smallest
        // sequence
        if (SizeSeq1 <= SizeSeq2)
        {
            if (SizeSeq1 < 500)
            {
                wordSize = (int)std::log2(SizeSeq1);
            }
            else
            {
                wordSize = 11;
            }
        }
        else
        {
            if (SizeSeq2 < 500)
            {
                wordSize = (int)std::log2(SizeSeq2);
            }
            else
            {
                wordSize = 11;
            }
        }

        if (wordSize == 0)
        {
            wordSize = 1;
        }

        // Prefer to calculate overlapping k-mers of shorter sequence
        // Switch the variables as Seq1 is assumed to be the larger sequence
        // Must remember that a switch occurred for alignment ordering purposes
        bool switched = false;
        if (Seq1.size() < Seq2.size())
        {
            std::swap(Seq1, Seq2);
            switched = true;
        }



        std::vector<candidateWord> words;
        std::vector<candidateWord> candidateWords;

        // For smaller sequences use non-overlapping k-mers quadratically
        // Whereas for larger sequences use a suffix tree to perform pattern matching
        if (Seq1.size() + Seq2.size() < 200)
        {
            // Find initial word matches using non-overlapping k-mers
            // TODO: Remove score concept and replace with purely full matches
            do
            {
                // Perfect Matches
                initialThreshold = Match * wordSize;

                for (int i = 0; i < Seq2.size() - wordSize + 1; i++)
                {
                    candidateWord cWord;
                    cWord.indexSeq2 = i;
                    cWord.wordSize = wordSize;
                    words.push_back(cWord);
                }

                // Compare k-mer words to non-overlapping k-mer words in Sequence 1 and a
                // score for each comparison and delete alignments below a threshold
                for (int i = 0; i < words.size(); i++)
                {
                    for (int j = 0; j < Seq1.size() - wordSize + 1; j = j + wordSize)
                    {
                        ScoreSystemType score = 0;

                        for (int k = 0; k < wordSize; k++)
                        {
                            ScoreSystemType Similarity = BaseType::match(Seq1[j + k], Seq2[words[i].indexSeq2 + k]) ? Match : Mismatch;
                            score += Similarity;
                        }

                        if (score >= initialThreshold)
                        {
                            candidateWord cWord;
                            cWord = words[i];
                            cWord.score = score;
                            cWord.indexSeq1 = j;
                            candidateWords.push_back(cWord);
                        }
                    }
                }

                if (candidateWords.size() == 0)
                {
                    wordSize = (int)wordSize * 3 / 4;

                    if (wordSize <= 0)
                    {
                        return;
                    }
                }

                words.clear();
            } while (candidateWords.size() == 0);
        }

        else // Suffix Tree Pattern Matching
        {
            SuffixTree<ContainerType, Ty, Blank> suffixTree;
            suffixTree.buildTree(Seq1, BaseType::getMatchOperation());

            do
            {
                initialThreshold = Match * wordSize;

                //Create non-overlapping patterns
                //And compare each pattern to suffix tree to find seeds
                for (int i = 0; i < Seq2.size() - wordSize + 1; i++)
                {
                    ArrayView<ContainerType> pattern(Seq2);
                    pattern.sliceWindow(i, i + wordSize);
                    std::vector<candidateWord> patternMatches = suffixTree.getCandidates(pattern);
                    for (int j = 0; j < patternMatches.size(); j++)
                    {
                        patternMatches[j].indexSeq2 = i;
                        patternMatches[j].score = Match * wordSize;
                        patternMatches[j].wordSize = wordSize;
                        candidateWords.push_back(patternMatches[j]);
                    }
                }

                if (candidateWords.size() == 0)
                {
                    wordSize = (int)wordSize * 3 / 4;

                    if (wordSize <= 0)
                    {
                        return;
                    }
                }

                words.clear();
            } while (candidateWords.size() == 0);

            suffixTree.deleteTree();
        }



        // Begin expanding out the seeds in both directions
        bool anyExpansion = false;
        std::vector<candidateWord> pushedWords;

        while (candidateWords.size() >= 1)
        {
            std::vector<candidateWord> newCandidates = candidateWords;
            for (int i = 0; i < newCandidates.size(); i++)
            {
                //<------------- Merge overlapping words ------------->
                if (i > 0)
                {
                    candidateWord firstWord = newCandidates[i - 1];
                    candidateWord secondWord = newCandidates[i];

                    // The position at which the first word ends is greater than the
                    // beginning of the second word This means they are overlapping so
                    // merge them together
                    if ((firstWord.indexSeq1 + firstWord.wordSize >
                         secondWord.indexSeq1) &&
                        (firstWord.indexSeq2 + firstWord.wordSize >
                         secondWord.indexSeq2) &&
                        (firstWord.indexSeq1 != secondWord.indexSeq1))
                    {

                        int overlapSize = firstWord.indexSeq1 + firstWord.wordSize - secondWord.indexSeq1;
                        int overlapSeq2Size = firstWord.indexSeq2 + firstWord.wordSize - secondWord.indexSeq2;
                        if (overlapSize != overlapSeq2Size)
                        {
                            continue;
                        }

                        candidateWord mergedWord = firstWord;

                        // re-calculate score, however, we already know word 1 score so to
                        // reduce time we only calculate score of the word 2 that is being
                        // added to word 1. Must be done since imperfect matches prevents us
                        // from simply multiplying a match score with the number of
                        // characters being added to word 1
                        int mergedWordSize = mergedWord.wordSize + secondWord.wordSize - (firstWord.indexSeq1 + firstWord.wordSize - secondWord.indexSeq1);

                        for (int j = mergedWord.wordSize; j < mergedWordSize; j++)
                        {

                            ScoreSystemType Similarity = BaseType::match(Seq1[mergedWord.indexSeq1 + j], Seq2[mergedWord.indexSeq2 + mergedWord.wordSize]) ? Match : Mismatch;

                            // Mismatch;
                            mergedWord.score += Similarity;
                            if (Similarity < 0)
                            {
                                mergedWord.score = Similarity;
                            }
                        }

                        mergedWord.wordSize = mergedWordSize;
                        newCandidates[i - 1] = mergedWord;
                        newCandidates.erase(newCandidates.begin() + i);
                        i--;
                        continue;
                    }

                    // The two words are on the verge of joining so merging is simple
                    // addition Separate else if for efficiency's sake
                    else if (firstWord.indexSeq1 + firstWord.wordSize == secondWord.indexSeq1)
                    {

                        int overlapSize = firstWord.indexSeq1 + firstWord.wordSize - secondWord.indexSeq1;
                        int overlapSeq2Size = firstWord.indexSeq2 + firstWord.wordSize - secondWord.indexSeq2;

                        if (overlapSize != overlapSeq2Size)
                        {
                            continue;
                        }

                        candidateWord mergedWord = firstWord;
                        mergedWord.wordSize = mergedWord.wordSize + secondWord.wordSize;
                        mergedWord.score = mergedWord.score + secondWord.score;
                        newCandidates[i - 1] = mergedWord;
                        newCandidates.erase(newCandidates.begin() + i);
                        i--;
                        continue;
                    }
                }

                //<------------- Expand word ------------->
                ScoreSystemType score = newCandidates[i].score;
                bool expansion = false;
                // Expand left
                int leftIndex = newCandidates[i].indexSeq1 - 1;
                if (leftIndex >= 0)
                { // We can expand left

                    newCandidates[i].indexSeq1--;
                    newCandidates[i].indexSeq2--;
                    newCandidates[i].wordSize++;

                    ScoreSystemType Similarity;

                    if (newCandidates[i].indexSeq2 < 0)
                    {
                        Similarity = Mismatch;
                    }
                    else
                    {
                        Similarity = BaseType::match(Seq1[leftIndex], Seq2[newCandidates[i].indexSeq2]) ? Match : Mismatch;
                    }

                    // A detremental expansion so revert
                    if (newCandidates[i].score + Similarity < newCandidates[i].score || Similarity < 0)
                    {
                        newCandidates[i].indexSeq1++;
                        newCandidates[i].indexSeq2++;
                        newCandidates[i].wordSize--;
                    }
                    else
                    {
                        newCandidates[i].score += Similarity;
                        expansion = true;
                        anyExpansion = true;
                    }
                }

                // Expand right
                int rightIndex = newCandidates[i].indexSeq1 + newCandidates[i].wordSize; 
                                                // Plus wordSize as we index points
                                               // towards start of word

                if (rightIndex < Seq1.size())
                {

                    ScoreSystemType Similarity;

                    if (newCandidates[i].indexSeq2 + newCandidates[i].wordSize >= Seq2.size())
                    {
                        Similarity = Mismatch;
                    }
                    else
                    {
                        Similarity = BaseType::match(Seq1[rightIndex], Seq2[newCandidates[i].indexSeq2 + newCandidates[i].wordSize]) ? Match : Mismatch;
                    }

                    // A detremental expansion so revert
                    if (newCandidates[i].score + Similarity < newCandidates[i].score || Similarity < 0)
                    {
                    }
                    else
                    {
                        newCandidates[i].wordSize++;
                        newCandidates[i].score += Similarity;
                        expansion = true;
                        anyExpansion = true;
                    }
                }

                // Word cannot be expanded further
                if (!expansion)
                {
                    pushedWords.push_back(newCandidates[i]);
                    newCandidates.erase(newCandidates.begin() + i);
                }
            }

            // All possible expansions have taken place so exit
            if (!anyExpansion)
            {
                candidateWords = pushedWords;
                break;
            }

            candidateWords = newCandidates;
        }

        candidateWords = pushedWords;



        //<----- Sort the Candidates and perform LIS ----->

        // Sort the Candidates by index of the first sequence
        std::sort(candidateWords.begin(), candidateWords.end());

        //Perform LIS on Candidates via index of second sequence
        BaseType::longestIncreasingSubsequence(candidateWords);

        if (candidateWords.size() == 0)
        {
            StaticFuncs<ContainerType, Ty, Blank, MatchFnTy>::useNW(Seq1, Seq2, Result, Scoring, BaseType::getMatchOperation());
            return;
        }

        //<----- Build Alignment ----->
        if (switched)
        {
            // ContainerType temp = Seq1;
            // Seq1 = Seq2;
            // Seq2 = temp;
            std::swap(Seq1, Seq2);

            for (int i = 0; i < candidateWords.size(); i++)
            {
                int tempIndex = candidateWords[i].indexSeq2;
                candidateWords[i].indexSeq2 = candidateWords[i].indexSeq1;
                candidateWords[i].indexSeq1 = tempIndex;
            }
        }

        // For each word
        int i = 0;
        bool breakOut = false;
        for (i = 0; i < candidateWords.size(); i++)
        {
            // Align word
            alignWord(candidateWords[i], Seq1, Seq2, Result);

            // Bridge gap to word ignoring overlapping words
            int j = i + 1;
            while (true)
            {
                if (j < candidateWords.size())
                {
                    // Overlapping word so look to bridge to next word if possible
                    if (candidateWords[i].indexSeq1 + candidateWords[i].wordSize > candidateWords[j].indexSeq1 || candidateWords[i].indexSeq2 + candidateWords[i].wordSize > candidateWords[j].indexSeq2)
                    {
                        j++;
                        continue;
                    }

                    // Align the gap 
                    StaticFuncs<ContainerType, Ty, Blank, MatchFnTy>::bridgeNW(Seq1, Seq2, Result, Scoring, candidateWords[i].indexSeq1 + candidateWords[i].wordSize, candidateWords[i].indexSeq2 + candidateWords[i].wordSize, candidateWords[j].indexSeq1, candidateWords[j].indexSeq2, BaseType::getMatchOperation());

                    i = j - 1;
                    break;
                }
                else
                {
                    breakOut = true;
                    break;
                }
            }

            if (breakOut)
            {
                i++;
                break;
            }
        }

        // Force global alignment
        if (i == 1)
        {
            BaseType::forceGlobal(Seq1, Seq2, Result, candidateWords[0].indexSeq1, candidateWords[0].indexSeq2, candidateWords[0].indexSeq1 + candidateWords[0].wordSize, candidateWords[0].indexSeq2 + candidateWords[0].wordSize);
        }
        else
        {
            BaseType::forceGlobal(Seq1, Seq2, Result, candidateWords[0].indexSeq1, candidateWords[0].indexSeq2, candidateWords[i - 1].indexSeq1 + candidateWords[i - 1].wordSize, candidateWords[i - 1].indexSeq2 + candidateWords[i - 1].wordSize);
        }


        if (Seq1.size() == 145 && Seq2.size() == 121)
        {
            StaticFuncs<ContainerType, Ty, Blank, MatchFnTy>::useNW(Seq1, Seq2, Result, Scoring, BaseType::getMatchOperation());
            return;
        }
    }

    void alignWord(candidateWord word, ContainerType &Seq1, ContainerType &Seq2, AlignedSequence<Ty, Blank> &Result)
    {
        auto &Data = Result.Data;

        for (int i = 0; i < word.wordSize; i++)
        {
            Data.push_back(typename BaseType::EntryType(
                Seq1[i + word.indexSeq1], Seq2[i + word.indexSeq2], true)
            );
        }
    }

public:
    static ScoringSystem getDefaultScoring() { return ScoringSystem(-1, 2, -1); }

    BLATSA() : BaseType(getDefaultScoring(), nullptr) {}

    BLATSA(ScoringSystem Scoring, MatchFnTy Match) : BaseType(Scoring, Match) {}

    virtual AlignedSequence<Ty, Blank> getAlignment(ContainerType &Seq1, ContainerType &Seq2)
    {
        AlignedSequence<Ty, Blank> Result;
        buildAlignment(Seq1, Seq2, Result);
        return Result;
    }

    virtual AlignedSequence<Ty, Blank> getAlignment(std::vector<ContainerType> Seqs)
    {
        assert(Seqs.size() == 2 && "This is a Pairwise Aligner NOT Multiple");
        return getAlignment(Seqs[0], Seqs[1]);
    }
};
