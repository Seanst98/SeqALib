
#include <algorithm>
#include <cassert>
#include <functional>
#include <limits.h> // INT_MIN
#include <list>
#include <iostream>
#include <fstream>
#include <vector>

#define ScoreSystemType int

#include "AlignmentVector.h"

// Store alignment result here
template <typename Ty, Ty Blank = Ty(0)>
class AlignedSequence
{
public:
    class Entry
    {
    private:

        std::vector<Ty> objects;  // Column of alignment - Could be two for pairwise, 3+ for multiple
                                  // Does open the ability to add just 1 but this doesn't make sense so be careful eh

        std::vector<bool> matches;  // Vector of matches where index 0 signifies a match/mismatch from the first character to the second
                                    // Index 1 signifies a match from second character to third
                                    // Will have a size one less than Column

    public:
        Entry() { matches = { false }; }

        // Pairwise
        // Assume matching when unknown
        Entry(Ty V1, Ty V2) : matches({ true }), objects({ V1, V2 }) {}

        Entry(Ty V1, Ty V2, bool Matching) : matches({ Matching }), objects({ V1, V2 }) {}

        // Multiple
        // Assume matching when unknown
        Entry(std::vector<Ty> V) : matches({ true }), objects(V) {}

        Entry(std::vector<Ty> V, std::vector<bool> B) : matches(B), objects(V) {}

        Entry(Entry entry, Ty V, bool Matching)
        {
            entry.add(V, Matching);
            objects = entry.objects;
            matches = entry.matches;
        }


        Ty get(size_t index)
        {
            assert((index >= 0 || index < objects.size()) && "Index out of bounds!");
            return objects[index];
        }

        bool empty()
        {
            for (int i = 0; i < objects.size(); i++)
            {
                if (objects[i] != Blank)
                    return false;
            }

            return true;
        }

        bool hasBlank()
        {
            for (int i = 0; i < objects.size(); i++)
            {
                if (objects[i] == Blank)
                    return true;
            }

            return false;
        }

        bool getMatch(size_t index)
        {
            return matches[index];
        }

        void add(Ty V, bool M)
        {
            objects.push_back(V);
            matches.push_back(M);
        }

    };

    //AlignVec<Entry> Data;
    std::list<Entry> Data;
    int nObjects = 2;  // Number of objects in an alignment column
                       // 2 as default (Pairwise)

    AlignedSequence()
    {
        //Data = AlignVec<Entry>(32);
    }

    AlignedSequence(int numObjects, size_t size) : nObjects(numObjects)
    {
        //Data = AlignVec<Entry>(size);
    }

    AlignedSequence(size_t size)
    {
        //Data = AlignVec<Entry>(size);
    }

    AlignedSequence(const AlignedSequence<Ty, Blank>& Other) : Data(Other.Data) {}
    //AlignedSequence(AlignedSequence<Ty, Blank> &&Other) : Data(std::move(Other.Data)) {}

    AlignedSequence(AlignedSequence<Ty, Blank> &Other, int numObjects) : Data(Other.Data), nObjects(numObjects) {}

    AlignedSequence<Ty> &operator=(const AlignedSequence<Ty, Blank> &Other)
    {
        Data = Other.Data;
        return (*this);
    }

    Entry& operator[](size_t index)
    {
        return Data[index];
    }

    void append(const AlignedSequence<Ty, Blank> &Other)
    {
        Data.insert(Data.end(), Other.Data.begin(), Other.Data.end());
    }

    void splice(AlignedSequence<Ty, Blank> &Other)
    {
        Data.splice(Data.end(), Other.Data);
        //Data.splice(Other.Data);
    }

    int size()
    {
        return Data.size();
    }

    void insert(Ty V, int index)
    {
        Data[index].add(Blank, false);
        auto entry = Entry(V, nObjects);
        Data.insert(Data.begin() + index, entry);
    }

    //typename std::list<Entry>::iterator begin() { return Data.begin(); }
    //typename std::list<Entry>::iterator end() { return Data.end(); }
    typename auto begin() { return Data.begin(); }
    typename auto end() { return Data.end(); }
};

class ScoringSystem
{
    ScoreSystemType Gap;
    ScoreSystemType Match;
    ScoreSystemType Mismatch;
    ScoreSystemType GapOpen;
    ScoreSystemType GapExtend;
    bool AllowMismatch;

public:
    ScoringSystem(ScoreSystemType Gap, ScoreSystemType Match)
    {
        this->Gap = Gap;
        this->Match = Match;
        this->Mismatch = std::numeric_limits<ScoreSystemType>::min();
        this->AllowMismatch = false;
    }

    ScoringSystem(ScoreSystemType Gap, ScoreSystemType Match,
                  ScoreSystemType Mismatch, bool AllowMismatch = true)
    {
        this->Gap = Gap;
        this->Match = Match;
        this->Mismatch = Mismatch;
        this->AllowMismatch = AllowMismatch;
    }

    ScoringSystem(ScoreSystemType GapOpen, ScoreSystemType GapExtend,
                  ScoreSystemType Match, ScoreSystemType Mismatch,
                  bool AllowMismatch = true)
    {
        this->GapOpen = GapOpen;
        this->GapExtend = GapExtend;
        this->Match = Match;
        this->Mismatch = Mismatch;
        this->AllowMismatch = AllowMismatch;
    }

    bool getAllowMismatch() { return AllowMismatch; }

    ScoreSystemType getMismatchPenalty() { return Mismatch; }

    ScoreSystemType getGapPenalty() { return Gap; }

    ScoreSystemType getMatchProfit() { return Match; }

    ScoreSystemType getGapOpenPenalty() { return GapOpen; }

    ScoreSystemType getGapExtendPenalty() { return GapExtend; }
};

template <typename ContainerType, typename Ty = typename ContainerType::value_type, Ty Blank = Ty(0), typename MatchFnTy = std::function<bool(Ty, Ty)>>
class SequenceAligner
{
private:
    ScoringSystem Scoring;
    MatchFnTy Match;

public:
    using EntryType = typename AlignedSequence<Ty, Blank>::Entry;

    SequenceAligner(ScoringSystem Scoring, MatchFnTy Match = nullptr) : Scoring(Scoring), Match(Match) {}

    ScoringSystem &getScoring() { return Scoring; }

    bool match(Ty Val1, Ty Val2) { return Match(Val1, Val2); }

    MatchFnTy getMatchOperation() { return Match; }

    Ty getBlank() { return Blank; }

    // Pairwise Alignment
    virtual AlignedSequence<Ty, Blank> getAlignment(ContainerType& Seq0, ContainerType& Seq1) = 0;

    // Multiple Alignment
    virtual AlignedSequence<Ty, Blank> getAlignment(std::vector<ContainerType> Seqs) = 0;

    // Force an alignment to be global
    void forceGlobal(ContainerType &Seq1, ContainerType &Seq2, AlignedSequence<Ty, Blank> &Result, int idx1, int idx2, int endIdx1, int endIdx2)
    {
        auto &Data = Result.Data;

        // Stick on front of sequences up to alignment
        AlignedSequence<Ty, Blank> front;
        for (int i = 0; i < idx1; i++)
        {
            front.Data.push_back(EntryType(Seq1[i], Blank, false));
        }
        for (int i = 0; i < idx2; i++)
        {
            front.Data.push_back(EntryType(Blank, Seq2[i], false));
        }

        front.splice(Result);

        // Stick on end of sequences
        AlignedSequence<Ty, Blank> end;
        for (int i = endIdx1; i < Seq1.size(); i++)
        {
            end.Data.push_back(EntryType(Seq1[i], Blank, false));
        }
        for (int i = endIdx2; i < Seq2.size(); i++)
        {
            end.Data.push_back(EntryType(Blank, Seq2[i], false));
        }

        front.splice(end);

        Result.Data.clear();

        Result.splice(front);
    }

    // Longest Increasing Subsequence Algorithm
    // Algorithms like MUMmer and BLAST make use of this
    // Might consider moving this to static funcs to reduce bloat
    template <typename ArrayType>
    void longestIncreasingSubsequence(ArrayType &array)
    {
        // Perform LIS on ArrayType via index of second sequence
        if (array.size() != 1)
        {
            // Denoting X as completeWords
            int* P = new int[array.size()];
            int* M = new int[array.size() + 1];

            int l = 0;
            int lo = 0;
            int hi = 0;
            for (int i = 0; i < array.size(); i++)
            {
                // Binary search for the largest positive j <= l
                // Such that X[M[j]] <= X[i]
                lo = 1;
                hi = l;
                while (lo <= hi)
                {
                    int mid = ceil((lo + hi) / 2);
                    if (array[M[mid]] <= array[i])
                    {
                        lo = mid + 1;
                    }
                    else
                    {
                        hi = mid - 1;
                    }
                }

                // After searching, lo is 1 greater than the length
                // of the longest prefix of X[i]
                int newL = lo;

                // The predecessor of X[i] is the last index of
                // the subsequence of length newL - 1
                P[i] = M[newL - 1];
                M[newL] = i;

                // If we found a subsequence longer than any we've found yet
                if (newL > l)
                {
                    l = newL;
                }
            }

            // Reconstruct the longest increasing subsequence
            ArrayType newList;
            newList.resize(l);
            int k = M[l];
            for (int i = l - 1; i >= 0; i--)
            {
                newList[i] = array[k];
                k = P[k];
            }

            array.clear();
            array = newList;
        }
    }
};

#include "ArrayView.h"
#include "SANeedlemanWunsch.h"
#include "staticFuncs.h"
#include "SAHirschberg.h"
#include "SASmithWaterman.h"
#include "SAGlobalGotoh.h"
#include "SALocalGotoh.h"
#include "SAMyersMiller.h"
#include "SAFOGSAA.h"
#include "CandidateWord.h"
#include "MUM.h"
#include "SuffixTree.h"
#include "SABLAT.h"
#include "SAMummer.h"
#include "MSAProgressiveNW.h"

#include "tsl/robin_growth_policy.h"
#include "tsl/robin_hash.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include "SearchStrategy.h"