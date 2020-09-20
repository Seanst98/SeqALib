#include <iostream>
#include <string>
#include <chrono>
#include <unordered_map>

#include "SequenceAlignment.h"

template <typename T>
bool equal(T V1, T V2) { return V1 == V2; }

template <typename Ty, Ty Blank>
void printAlignment(AlignedSequence<Ty, Blank> &Result)
{
    for (int i = 0; i < Result.nObjects - 1; i++)
    {
        for (auto& Entry : Result)
        {
            std::cout << Entry.get(i);
        }
        std::cout << std::endl;

        for (auto& Entry : Result)
        {
            if (Entry.getMatch(i))
                std::cout << "|";
            else
                std::cout << " ";
        }
        std::cout << std::endl;
    }

    for (auto& Entry : Result)
    {
        std::cout << Entry.get(Result.nObjects-1);
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{

    std::string seq1 = "AATCG";
    std::string seq2 = "AACG";
    std::string seq3 = "GAATG";
    //std::string seq1 = "AGGATCGGCTAGAGCTAGAGCTAGCTAGTAGC";
    //std::string seq2 = "GAGATCGGCGGATTACAGGCTATCGA";
    //std::string seq1 = "AAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    //std::string seq2 = "AAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    //std::string seq1 = "AAAAGGGGTTTTCCCC";
    //std::string seq2 = "AAAAGGGGTTTTCCCC";
    //std::string seq1 = "AAAGGGTTTCCC";
    //std::string seq2 = "AAAGGGTTTCCC";
    //std::string seq1 = "AAAA";
    //std::string seq2 = "AAAA";
    //std::string seq1 = "CTGAAGCGG";
    //std::string seq2 = "CTCAAGCGTAGTCC";
    //std::string seq1 = "AGTAC";
    //std::string seq2 = "AAG";
    //std::string seq1 = "GGTCGCGACTACGTGAGCTAGGGCTCCGGACTGGGCTGTATAGTCGAGTC";
    //std::string seq2 = "TGATCTCGCCCCGACAACTGCAAACCCCAACTTATTTAGATAACATGGTTTACAGG";
    //std::string seq1 = "AGGGGGGGGTTTTCAAAAGCTCTTCGCATGCGCATAGGCTTAGGAGTCGAGGATCGGCGGCATATTAAGAGAGGGCGGCGCCCATATATTAGGCGCGCGCGATTATATTATAATATATATATTAATCGGGCGCTATGCGC";
    //std::string seq2 = "AGGTCTCCGGCATGAGCTAGCGGCGAGTTATAGCGCGTTTCAAAAGCTCTTCGCATGCGCATAGGCTTAGGAGTCGAGGATCGGCGGCATATTAAGAGAGGGCGGCAGGCTTGGAAAAAGGCTCTGAGATGCGAGAATAGAGGAGAG";
    //std::string seq1 = "GTAGATTGCCATGCGTAGAGCTAACGAGCCAGCGGAAAGCGTGAGGCGCTTTTAAGCATGGCGAGTAAGTGATCCAACGCTTCGGATATGACTATATACTTAGGTTCGATCTCGTCCCGAGAATTCTAAGCCTCAACATCTATGAGTTATGAGGTTAGCCGAAAAAGCACGTGGTGGCGCCCACCGACTGTTCCCAGACTGTAGCTCTTTGTTCTGTCAAGGCCCGACCTTCATCGCGGCCGATTCCTTCTGCGGACCATACCGTCCTGATACTTTGGTCATGTTTCCGTTGTAGGAGTGAACCCACTTGCCTTTGCGTCTTAATACCAATGAAAAACCTATGCACTTTGTACAGGGTACCATCGGGATTCTGAACCCTCAGATAGTGGGGATCCCGGGTATAGACCTTTATCTGCGGTCCAACTTAGGCATAAACCTCCATGCTACCTTGTCAGACCCACCCTGCACGAGGTAAATATGGGACGCGTCCGACCTGGCTCCTGGCGTTCTACGCCGCCACGTGTTCGTTAACTGTTGATTGGTAGCACAAAAGTAATACCATGGTCCTTGAAATTCGGCTCAGTTAGTTCGAGCGTAATGTCACAAATGGCGCAGAACGGCAATGAGTGTTTGACACTAGGTGGTGTTCAGTTCGGTAACGGAGAGACTGTGCGGCATACTTAATTATACATTTGAAACGCGCCCAAGTGACGCTAGGCAAGTCAGAGCAGGTTCCCGTGTTAGCTTAAGGGTAAACATACAAGTCGATTGAAGATGGGTAGGGGGCTTCAATTCGTCCAGCACTCTACGGTACCTCCGAGAGCAAGTAGGGCACCCTGTAGTTCGAAGCGGAACTATTTCGTGGGGCGAGCCCACATCGTCTCTTCTGCGGATGACTTAACACGTTAGGGAGGTGGAGTTGATTCGAACGATGGTTATAAATCAAAAAAACGGAACGCTGTCTGGAGGATGAATCTAACGGTGCGTAACTCGATCACTCGTAGATTGCCATGCGTAGAGCTAACGAGCCAGCGGAAAGCGTGAGGCGCTTTTAAGCATGGCGAGTAAGTGATCCAACGCTTCGGATATGACTATATACTTAGGTTCGATCTCGTCCCGAGAATTCTAAGCCTCAACATCTATGAGTTATGAGGTTAGCCGAAAAAGCACGTGGTGGCGCCCACCGACTGTTCCCAGACTGTAGCTCTTTGTTCTGTCAAGGCCCGACCTTCATCGCGGCCGATTCCTTCTGCGGACCATACCGTCCTGATACTTTGGTCATGTTTCCGTTGTAGGAGTGAACCCACTTGCCTTTGCGTCTTAATACCAATGAAAAACCTATGCACTTTGTACAGGGTACCATCGGGATTCTGAACCCTCAGATAGTGGGGATCCCGGGTATAGACCTTTATCTGCGGTCCAACTTAGGCATAAACCTCCATGCTACCTTGTCAGACCCACCCTGCACGAGGTAAATATGGGACGCGTCCGACCTGGCTCCTGGCGTTCTACGCCGCCACGTGTTCGTTAACTGTTGATTGGTAGCACAAAAGTAATACCATGGTCCTTGAAATTCGGCTCAGTTAGTTCGAGCGTAATGTCACAAATGGCGCAGAACGGCAATGAGTGTTTGACACTAGGTGGTGTTCAGTTCGGTAACGGAGAGACTGTGCGGCATACTTAATTATACATTTGAAACGCGCCCAAGTGACGCTAGGCAAGTCAGAGCAGGTTCCCGTGTTAGCTTAAGGGTAAACATACAAGTCGATTGAAGATGGGTAGGGGGCTTCAATTCGTCCAGCACTCTACGGTACCTCCGAGAGCAAGTAGGGCACCCTGTAGTTCGAAGCGGAACTATTTCGTGGGGCGAGCCCACATCGTCTCTTCTGCGGATGACTTAACACGTTAGGGAGGTGGAGTTGATTCGAACGATGGTTATAAATCAAAAAAACGGAACGCTGTCTGGAGGATGAATCTAACGGTGCGTAACTCGATCACTC";
    //std::string seq2 = "ACTCGCTATTCGAACTGCGCGAAAGTTCCCAGCGCTCATACACTTGGTTCCGAGGCCTGTCCTGATATATGAACCCAAACTAGAGCGGGGCTGTTGACGTTTGGAGTTGAAAAAATCTAATATTCCAATCGGCTTCAACGTGCACCACCGCAGGCGGCTGACGAGGGGCTCACACCGAGAAAGTAGACTGTTGCGCGTTGGGGGTAGCGCCGGCTAACAAAGACGCCTGGTACAGCAGGAGTATCAAACCCGTACAAAGGCAACATCCTCACTTCGGTGAATCGAAGCGCGGCATCAGGGTTACTTTTTGGATACCTGAAACAAAACCCATCGTAGTCCTTAGACTTGGCACACTTACACCTGCAGCGCGCGCATCTGGAAATAGAGGCCAAGTTCGATCCGTACTCCGACGTACGATGCAACAGTGTGGATGTGACGAGCTTCATTTATACCCTTCGCGCGCCGGACCGGCCTCCGCAAGGCGCGGCGGTGCACAAGCAATTGACAACTAACCACCGTGTATTCGTTATGGCATCAGGCAGTTTAAGTCGAGACAATAGGGCTCGCAATACACAGTTTACCGCATCTTGCCCTAACTGACAAACTGTGATCGACCACTAGCCATGCCATTGCCTCTTAGACACCCCGATACAGTGATTATGAAAGGTTTGCGGGGCATGGCTACGACTTGTTCAGCTACGTCCGAGGGCAGAAACTTATCCCCATTTGTATGTTCACCTATCTACTACCCATCCCCGGAGGTTAAGTAGGTTGTGAGATGCGGGAGAGGTTCTCGATCTTCCCGTGGGACGTCAACCTTTCCCTTGATAAAGCATCCCGCTCGGGTATGGCAGTGAGTACGCCTTCTGAATTGTGCTATCCTTCGTCCTTATCAAAGCTTGCTACCAATAATTAGGATTATTGCCTTGCGACAGACTTCCTACTCACACTCCCTCACATTGAGCTACTCGATGGGCGATTAGCTTGACCCGCTCTGTAGACTCGCTATTCGAACTGCGCGAAAGTTCCCAGCGCTCATACACTTGGTTCCGAGGCCTGTCCTGATATATGAACCCAAACTAGAGCGGGGCTGTTGACGTTTGGAGTTGAAAAAATCTAATATTCCAATCGGCTTCAACGTGCACCACCGCAGGCGGCTGACGAGGGGCTCACACCGAGAAAGTAGACTGTTGCGCGTTGGGGGTAGCGCCGGCTAACAAAGACGCCTGGTACAGCAGGAGTATCAAACCCGTACAAAGGCAACATCCTCACTTCGGTGAATCGAAGCGCGGCATCAGGGTTACTTTTTGGATACCTGAAACAAAACCCATCGTAGTCCTTAGACTTGGCACACTTACACCTGCAGCGCGCGCATCTGGAAATAGAGGCCAAGTTCGATCCGTACTCCGACGTACGATGCAACAGTGTGGATGTGACGAGCTTCATTTATACCCTTCGCGCGCCGGACCGGCCTCCGCAAGGCGCGGCGGTGCACAAGCAATTGACAACTAACCACCGTGTATTCGTTATGGCATCAGGCAGTTTAAGTCGAGACAATAGGGCTCGCAATACACAGTTTACCGCATCTTGCCCTAACTGACAAACTGTGATCGACCACTAGCCATGCCATTGCCTCTTAGACACCCCGATACAGTGATTATGAAAGGTTTGCGGGGCATGGCTACGACTTGTTCAGCTACGTCCGAGGGCAGAAACTTATCCCCATTTGTATGTTCACCTATCTACTACCCATCCCCGGAGGTTAAGTAGGTTGTGAGATGCGGGAGAGGTTCTCGATCTTCCCGTGGGACGTCAACCTTTCCCTTGATAAAGCATCCCGCTCGGGTATGGCAGTGAGTACGCCTTCTGAATTGTGCTATCCTTCGTCCTTATCAAAGCTTGCTACCAATAATTAGGATTATTGCCTTGCGACAGACTTCCTACTCACACTCCCTCACATTGAGCTACTCGATGGGCGATTAGCTTGACCCGCTCTGTAG";
    //std::string seq1 = "ACGGTTGC";
    //std::string seq2 = "AGCGTC";
    //std::string seq1 = "AGTC";
    //std::string seq2 = "GACT";
    //std::string seq1 = "AAGG";
    //std::string seq2 = "AAGG";
    //std::string seq1 = "AAGGGGGCAACCCAATTGTCAAAA";
    //std::string seq2 = "AAGTTGGCGGCCCAAGCTGCGAAA";
    //std::string seq1 = "AAAGGGTTTCCCAAGGTTCCAGTC";
    //std::string seq2 = "AAAGGGTTTCCCAAGGCTCCAGTC";
    //std::string seq1 = "CGAA";
    //std::string seq2 = "CGA";
    //std::string seq1 = "AGCTTCAGGCTGA";
    //std::string seq2 = "AGCTGGATCGATCGATG";
    //std::string seq1 = "AGCTCGATCAA";
    //std::string seq2 = "GCAACCGATCGA";
    //std::string seq1 = "AAAGAATGCAT";
    //std::string seq2 = "AAACTCAT";
    //std::string seq1 = "AA";
    //std::string seq2 = "A";
    //std::string seq1 = "AATCGG";
    //std::string seq2 = "AACGGT";

    // AAA GAA TGCAT
    // | |  |  | |||
    // A A  A CT CAT

    // AAA GAATGCAT
    // |||    | |||
    // AAAC   T CAT

    if (argc > 1)
    {
        seq1 = std::string(argv[1]);
        seq2 = std::string(argv[2]);
    }

    ScoringSystem scoringSystem = ScoringSystem(-2, 1, -1, false);
    ScoringSystem affineSystem = ScoringSystem(-3, -1, 1, -1, false);

    auto t1 = std::chrono::high_resolution_clock::now();
    NeedlemanWunschSA<std::string, char, '-'> SA(scoringSystem, equal<char>); //could also use `nullptr` instead of `equal<char>`
    AlignedSequence<char, '-'> NWAlignment = SA.getAlignment(seq1, seq2);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "# Needleman-Wunsch:" << std::endl;
    printAlignment(NWAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t3 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> HAlignment = HirschbergSA<std::string, char, '-'>(scoringSystem, equal<char>).getAlignment(seq1, seq2);
    auto t4 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    std::cout << "# Hirschberg:" << std::endl;
    printAlignment(HAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t5 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> SWAlignment = SmithWatermanSA<std::string, char, '-'>(scoringSystem, equal<char>).getAlignment(seq1, seq2);
    auto t6 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
    std::cout << "# Smith-Waterman:" << std::endl;
    printAlignment(SWAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t7 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> LGAlignment = LocalGotohSA<std::string, char, '-'>(affineSystem, equal<char>).getAlignment(seq1, seq2);
    auto t8 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();
    std::cout << "# Local Gotoh:" << std::endl;
    printAlignment(LGAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t9 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> GGAlignment = GlobalGotohSA<std::string, char, '-'>(affineSystem, equal<char>).getAlignment(seq1, seq2);
    auto t10 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t10 - t9).count();
    std::cout << "# Global Gotoh:" << std::endl;
    printAlignment(GGAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t11 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> MMAlignment = MyersMillerSA<std::string, char, '-'>(affineSystem, equal<char>).getAlignment(seq1, seq2);
    auto t12 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();
    std::cout << "# Myers Miller: " << std::endl;
    printAlignment(MMAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t15 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> FOGSAAAlignment = FOGSAASA<std::string, char, '-'>(scoringSystem, equal<char>).getAlignment(seq1, seq2);
    auto t16 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t16 - t15).count();
    std::cout << "# FOGSAA: " << std::endl;
    printAlignment(FOGSAAAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t17 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> BLATAlignment = BLATSA<std::string, char, '-'>(scoringSystem, equal<char>).getAlignment(seq1, seq2);
    auto t18 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t18 - t17).count();
    std::cout << "# BLAT: " << std::endl;
    printAlignment(BLATAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t23 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> MummerAlignment = MummerSA<std::string, char, '-'>(scoringSystem, equal<char>).getAlignment(seq1, seq2);
    auto t24 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t24 - t23).count();
    std::cout << "# MUMMER: " << std::endl;
    printAlignment(MummerAlignment);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t21 = std::chrono::high_resolution_clock::now();
    SuffixTree<std::string, char, '-'> suffixTree;
    AlignedSequence<char, '-'> suffixTreeLCS = suffixTree.getLCS(seq1, seq2, SA.getMatchOperation());
    auto t22 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t22 - t21).count();
    std::cout << "# Suffix Tree LCS: " << std::endl;
    printAlignment(suffixTreeLCS);
    suffixTree.deleteTree();
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    auto t25 = std::chrono::high_resolution_clock::now();
    std::vector<candidateWord> words;
    std::string pattern = "GEEK";
    //std::string text = "GEEKSFORGEEKS";
    std::string text = seq1;
    SuffixTree<std::string> suffTree;
    suffTree.buildTree(text, SA.getMatchOperation());
    words = suffTree.getCandidates(pattern);
    auto t26 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t26 - t25).count();
    std::cout << "# Suffix Tree Pattern Matches: " << std::endl;
    std::cout << "# Finding Matches of " << pattern << " in " << text << std::endl;
    if (words.size() == 0)
    {
        std::cout << pattern << " is not a substring of " << text << std::endl;
    }
    for (int i = 0; i < words.size(); i++)
    {
        for (int j = 0; j < pattern.size(); j++)
        {
            std::cout << text[words[i].indexSeq1 + j];
        }
        std::cout << std::endl;
    }
    suffTree.deleteTree();
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << "" << std::endl;

    std::cout << std::endl;
    std::cout << "<------------------------------------------------------------------------->" << std::endl;

    std::cout << std::endl;
    double similarity;
    SearchStrategy<std::string, char, '-'> searchStrategy(SA.getMatchOperation());

    std::ifstream file("random_strings.txt");
    std::vector<std::string> strings;

    std::string str;
    std::string myString = "gunMXMmwikiuptglaoKhyIBySaaFcTCHqZjOCpKaQRXDBkeNVrZChMbZnDHtPIHlICZyMiBsyYEZFwdNkSpnQsinVsunWjBmXWMgBxJcZSkxTNTQgqJpsRSMlrTKiCJeTWpsNRrzdJGBmLQuZKbidfrGKPStCbmzhcbVfrrsSWMrjCGUwpdVpljqIHlTYaRKfnHnzzBlSumsiwvLEdMrcLiqmEmsVWRzTvmrnseYGQQQUVornHlBWNBzfafZvtJYVsxaZbVfskwCkdSToGAVnVPcZoyOxWrWgABNdjXpEXqTNVDQcMmyqjcoxeftUHJGhYuuZRzeZLBeIcEBKHhcspZzIbqVvnvZkbZIJZldvmFqRBQpPuPuTwFrjNWoJSJIfsjwtHSHwlQnOmGlqfYsArgwJxRuzjaWQeFfFfFYNsrdiUBkCFTQQCdmDALEoskNxTqOwrGHXUkPOvYPQPRiEYWwApRYgMNVPcEMMzEycjxmSBzcvBDNCmruxQlKYNRhOZaKEPqZacklcmlFxdTjRMqSrMafrIfGdZczMHtZCfyIsmNsjNnLESkyMzKszAlEGUPdUXmhFBJNkZDTVHxNDzjuyCqLFkdGDEqSMPjOpjSuRRZXffwlCbBRWgPQFvjKXhnPvofyiXwkRWMEQnWfTmObJtEcGuUkvQhFjQoegSokXDdYHKdTUHjhQGHyecmRYUvkOMbuHaUGWSjRqItuNBiQvdCElaqKFTkkXXYhxTLggbeeBPLOjRpNHFcLLUFDpJxXzJxfAyrEBnyOQLGcNSDrtyjRqNavnRLJYDJPMolpwxjxvLBqgcxHfDNBqikbSNzYWhsJjItvXetOOqVSgNlqiUyHuuKRoyxiCJuUddzfkaJajgRXILqsxPHBcagyaZldBggcRtcSqLNbrqUsQqRAxzdZPzpwVglSyhcMBmJatUxNIZlsqRBsBfvFzFRzJauycIrmoTdejQrbvWiVxiDZUJAyhJxPTXMLvcZo";

    while (std::getline(file, str))
    {
        if (str.size() > 0)
            strings.push_back(str);
    }

    std::cout << strings.size() << std::endl;
    double maxScore = 0.0;
    int index = 0;
    uint32_t nHashes = 200;


    auto t27 = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> randomHashFuncs(nHashes - 1);
    searchStrategy.generateRandomHashFunctions(nHashes - 1, randomHashFuncs);

    std::vector<uint32_t> shingleHashesSeq1(nHashes);
    std::vector<uint32_t> shingleHashesSeq2(nHashes);
    searchStrategy.generateShinglesMultipleHashPipelineTurbo<5>(myString, nHashes, shingleHashesSeq1, randomHashFuncs);

    for (int i = 0; i < strings.size(); i++)
    {
        searchStrategy.generateShinglesMultipleHashPipelineTurbo<5>(strings[i], nHashes, shingleHashesSeq2, randomHashFuncs);
        double temp = searchStrategy.JaccardSingleHashFast(shingleHashesSeq1, shingleHashesSeq2, 0.8);

        if (temp > maxScore)
        {
            maxScore = temp;
            index = i;
        }
    }

    auto t28 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Most similar string found at index: " << index << " which is string: " << strings[index] << std::endl;


    duration = std::chrono::duration_cast<std::chrono::microseconds>(t28 - t27).count();   

    std::cout << "# Similarity Score: " << std::endl;

    std::cout << "Similarity: " << maxScore << std::endl;

    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;

    std::cout << std::endl;






    std::cout << "# Pre-Computation For LSH: " << std::endl;

    auto t29 = std::chrono::high_resolution_clock::now();

    uint32_t rows = 10;
    uint32_t bands = 20;

    struct LSH {
        std::vector<tsl::robin_map<uint32_t, std::string*>> bands;
    }lsh;

    lsh.bands.resize(bands);
    for (int i = 0; i < bands; i++)
    {
        lsh.bands[i].reserve(strings.size());
    }

    std::vector<uint32_t> bandHashes(bands);
    std::vector<uint32_t> shingleHashes(nHashes);



    // For each string
    for (int i = 0; i < strings.size(); i++)
    {
        searchStrategy.generateShinglesMultipleHashPipelineTurbo<5>(strings[i], nHashes, shingleHashes, randomHashFuncs);
        searchStrategy.generateBands(shingleHashes, rows, bands, bandHashes);

        for (int j = 0; j < bands; j++)
        {
            std::string* ptrStr = &strings[i];
            lsh.bands[j].insert(std::make_pair(bandHashes[j], ptrStr));
        }
    }

    auto t30 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t30 - t29).count();
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;

    std::cout << std::endl;










    maxScore = 0.0;
    std::string finalFoundStr;

    auto t31 = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> myStringHashes; myStringHashes.resize(nHashes);
    searchStrategy.generateShinglesMultipleHashPipelineTurbo<5>(myString, nHashes, myStringHashes, randomHashFuncs);

    std::vector<uint32_t> myStringBands; myStringBands.resize(bands);
    searchStrategy.generateBands(myStringHashes, rows, bands, myStringBands);

    std::vector<uint32_t> foundStrHashes; foundStrHashes.resize(nHashes);

    for (int i = 0; i < bands; i++)
    {
        if (lsh.bands[i].count(myStringBands[i]) > 0)
        {
            // We have a match so find it and record similarity
            std::cout << "FOUND A MATCH" << std::endl;
            std::string foundStr = *lsh.bands[i].at(myStringBands[i]);

            searchStrategy.generateShinglesMultipleHashPipelineTurbo<5>(foundStr, nHashes, foundStrHashes, randomHashFuncs);

            double temp = searchStrategy.JaccardSingleHashFast(myStringHashes, foundStrHashes, 0.8);

            std::cout << lsh.bands[i].count(myStringBands[i]) << std::endl;
            std::cout << temp << std::endl;

            if (temp > maxScore)
            {
                maxScore = temp;
                finalFoundStr = foundStr;
            }
        }
    }


    auto t32 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t32 - t31).count();

    std::cout << "# Find Most Similar: " << std::endl;
    std::cout << "Most Similar String: " << finalFoundStr << std::endl;
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;






    //std::cout << "Similarity: " << similarity << std::endl;

    /*std::vector<std::string> Seqs = { seq1, seq2, seq3 };
    auto t27 = std::chrono::high_resolution_clock::now();
    AlignedSequence<char, '-'> MSANW = NeedlemanWunschMSA<std::string, char, '-'>(scoringSystem, equal<char>).getAlignment(Seqs);
    auto t28 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t28 - t27).count();
    std::cout << "# MSANW: " << std::endl;
    printAlignment(MSANW);
    std::cout << "Time Taken: " << duration << " microseconds" << std::endl;
    std::cout << std::endl;*/


    //std::getchar();

    return 0;
}

uint32_t fakeHash(uint32_t key)
{
    return key;
}
