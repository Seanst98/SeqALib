#include <functional>
#include <algorithm>
#include <unordered_set>
#include <random>
#include <execution>
#include <thread>

template <typename ContainerType, typename Ty = typename ContainerType::value_type, Ty Blank = Ty(0), typename MatchFnTy = std::function<bool(Ty, Ty)>>
class SearchStrategy
{
private:

    MatchFnTy matchFn;

    std::vector<uint32_t> hashes;

public:

    SearchStrategy(MatchFnTy match) : matchFn(match) {}


    uint32_t fnv1a(const ContainerType &Seq)
    {
        uint32_t hash = 2166136261;
        int len = Seq.size();

        for (int i = 0; i < len; i++)
        {
            hash ^= Seq[i];
            hash *= 1099511628211;
        }

        return hash;
    }

    uint32_t fnv1a(const std::vector<uint32_t> &Seq)
    {
        uint32_t hash = 2166136261;
        int len = Seq.size();

        for (int i = 0; i < len; i++)
        {
            hash ^= Seq[i];
            hash *= 1099511628211;
        }

        return hash;
    }


    template<uint32_t K>
    std::vector<uint32_t>& generateShinglesSingleHashPipelineTurbo(const ContainerType &Seq, uint32_t nHashes, std::vector<uint32_t> &ret)
    {
        uint32_t pipeline[K] = { 0 };
        int len = Seq.size();

        std::unordered_set<uint32_t> set;
        //set.reserve(nHashes);
        uint32_t last = 0;



        for (int i = 0; i < len; i++)
        {

            for (int k = 0; k < K; k++)
            {
                pipeline[k] ^= Seq[i];
                pipeline[k] *= 1099511628211;
            }

            //Collect head of pipeline
            if (last <= nHashes-1)
            {
                ret[last++] = pipeline[0];
                
                if (last > nHashes - 1)
                {
                    std::make_heap(ret.begin(), ret.end());
                    std::sort_heap(ret.begin(), ret.end());
                }
            }

            if (pipeline[0] < ret.front() && last > nHashes-1)
            {
               if (set.find(pipeline[0]) == set.end())
                {
                    set.insert(pipeline[0]);

                    ret[last] = pipeline[0];

                    std::sort_heap(ret.begin(), ret.end());
                }
            }

            //Shift pipeline
            for (int k = 0; k < K - 1; k++)
            {
                pipeline[k] = pipeline[k + 1];
            }
            pipeline[K - 1] = 2166136261;
        }

        return ret;
    }

    /*template<uint32_t K>
    std::vector<uint32_t>& generateShinglesMultipleHashPipelineTurbo(const ContainerType& Seq, uint32_t nHashes, std::vector<uint32_t>& ret, std::vector<uint32_t>& ranHash)
    {
        uint32_t pipeline[K] = { 0 };
        int len = Seq.size();

        uint32_t smallest = std::numeric_limits<uint32_t>::max();

        std::vector<uint32_t> shingleHashes(len);

        // Pipeline to hash all shingles using fnv1a
        // Store all hashes
        // While storing smallest
        // Then for each shingle hash, rehash with an XOR of 32 bit random number and store smallest
        // Do this nHashes-1 times to obtain nHashes minHashes quickly
        // Sort the hashes at the end

        for (int i = 0; i < len; i++)
        {
            for (int k = 0; k < K; k++)
            {
                pipeline[k] ^= Seq[i];
                pipeline[k] *= 1099511628211;
            }

            //Collect head of pipeline
            if (pipeline[0] < smallest)
            {
                smallest = pipeline[0];
            }
            shingleHashes[i] = pipeline[0];

            //Shift pipeline
            for (int k = 0; k < K - 1; k++)
            {
                pipeline[k] = pipeline[k + 1];
            }
            pipeline[K - 1] = 2166136261;
        }

        ret[0] = smallest;

        // Now for each hash function, rehash each shingle and store the smallest each time
        for (int i = 0; i < ranHash.size(); i++)
        {
            smallest = std::numeric_limits<uint32_t>::max();

            for (int j = 0; j < shingleHashes.size(); j++)
            {
                uint32_t temp = shingleHashes[j] ^ ranHash[i];
                
                if (temp < smallest)
                {
                    smallest = temp;
                }
            }

            ret[i+1] = smallest;
        }

        std::sort(ret.begin(), ret.end());

        return ret;
    }*/

    template<uint32_t K>
    std::vector<uint32_t>& generateShinglesMultipleHashPipelineTurbo(const ContainerType& Seq, uint32_t nHashes, std::vector<uint32_t>& ret, std::vector<uint32_t>& ranHash)
    {
        uint32_t pipeline[K] = { 0 };
        int len = Seq.size();

        uint32_t smallest = std::numeric_limits<uint32_t>::max();

        std::vector<uint32_t> shingleHashes(len);

        // Pipeline to hash all shingles using fnv1a
        // Store all hashes
        // While storing smallest
        // Then for each shingle hash, rehash with an XOR of 32 bit random number and store smallest
        // Do this nHashes-1 times to obtain nHashes minHashes quickly
        // Sort the hashes at the end

        for (int i = 0; i < len; i++)
        {
            for (int k = 0; k < K; k++)
            {
                pipeline[k] ^= Seq[i];
                pipeline[k] *= 1099511628211;
            }

            //Collect head of pipeline
            if (pipeline[0] < smallest)
            {
                smallest = pipeline[0];
            }
            shingleHashes[i] = pipeline[0];

            //Shift pipeline
            for (int k = 0; k < K - 1; k++)
            {
                pipeline[k] = pipeline[k + 1];
            }
            pipeline[K - 1] = 2166136261;
        }

        ret[0] = smallest;

        // Now for each hash function, rehash each shingle and store the smallest each time
        // BUT THIS TIME USING THREADS OOOOOOO

        constexpr size_t numThreads = 5;
        constexpr size_t dataChunk = 200 / numThreads;
        constexpr size_t size = numThreads * dataChunk;


        // Declare thread pool
        std::thread* pool[numThreads];

        // Declare each threads entry function
        auto findSmallestXor = [&](size_t offset) -> void {
            //For this threads set of hashes
            std::vector<uint32_t> newHashes = shingleHashes;
            for (size_t i = offset; i < offset + dataChunk; ++i)
            {
                for (int j = 0; j < newHashes.size(); j++)
                {
                    newHashes[j] = shingleHashes[j] ^ ranHash[i];
                }

                //std::cout << i << std::endl;

                auto min = std::min_element(newHashes.begin(), newHashes.end());
                ret[i + 1] = newHashes[min - newHashes.begin()];
                //std::cout << ret[i + 1] << std::endl;
            }
        };

        // Spawn all threads, passing (i * dataChunk) as first parameter
        for (int i = 0; i < numThreads; ++i) {
            pool[i] = new std::thread(findSmallestXor, i * dataChunk);
        }

        // Wait for all threads to finish execution and delete them
        for (int i = 0; i < numThreads; ++i) {
            pool[i]->join();
            delete pool[i];
        }

        // Print results
        /*for (int i = 0; i < size; ++i) {
            std::cout << ret[i] << "\n";
        }*/


        std::sort(ret.begin(), ret.end());

        return ret;
    }

    constexpr std::vector<uint32_t>& generateRandomHashFunctions(int num, std::vector<uint32_t>& ret)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<> distribution(0, std::numeric_limits<uint32_t>::max());

        //generating a random integer:
        for (int i = 0; i < num; i++)
        {
            ret[i] = distribution(gen);
        }
        return ret;
    }

    std::vector<uint32_t>& generateBands(const std::vector<uint32_t> &minHashes, uint32_t rows, uint32_t bands, std::vector<uint32_t> &lsh)
    {

        // Generate a hash for each band
        for (int i = 0; i < bands; i++)
        {
            // Perform fnv1a on the rows
            auto first = minHashes.begin() + (i*rows);
            auto last = minHashes.begin() + (i*rows) + rows;
            lsh[i] = fnv1a(std::vector<uint32_t>{first, last});
        }

        return lsh;
    }

    double JaccardSingleHashFast(const std::vector<uint32_t> &seq1, const std::vector<uint32_t> &seq2, double alpha)
    {
        int len1 = seq1.size();
        int len2 = seq2.size();
        int nintersect = 0;
        int pos1 = 0;
        int pos2 = 0;
        int s = 0;

        const int smax = (int)std::ceil((1.0 - alpha) / (1.0 + alpha) * (len1 + len2));

        while (pos1 < len1 && pos2 < len2)
        {
            if (seq1[pos1] == seq2[pos2])
            {
                nintersect++;
                pos1++;
                pos2++;
            }
            else if (seq1[pos1] < seq2[pos2])
            {
                pos1++;
                s++;
            }
            else {
                pos2++;
                s++;
            }

            if (s > smax)
            {
                return 0.0;
            }
        }

        int nunion = len1 + len2 - nintersect;
        return nintersect / (double)nunion;
    }


    /*template<uint32_t K>
    std::vector<uint32_t> generateShinglesSingleHashPipeline(const ContainerType &Seq)
    {
        uint32_t pipeline[K] = { 0 };
        int len = Seq.size();
        std::vector<uint32_t> ret(len);

        for (int i = 0; i < len; i++)
        {

            for (int k = 0; k < K; k++)
            {
                pipeline[k] ^= Seq[i];
                pipeline[k] *= 1099511628211;
            }

            //Collect head of pipeline
            ret[i] = pipeline[0];

            //Shift pipeline
            for (int k = 0; k < K - 1; k++)
            {
                pipeline[k] = pipeline[k + 1];
            }
            pipeline[K - 1] = 2166136261;
        }

        std::partial_sort(ret.begin(), ret.begin() + 200, ret.end());
        auto first = ret.begin();
        auto last = ret.begin() + 200;

        std::vector<uint32_t> newVec(first, last);

        return newVec;
    }*/

    /*template<uint32_t K>
    std::vector<uint32_t> generateShinglesSingleHashPipelinePrem(ContainerType Seq)
    {
        uint32_t pipeline[K] = { 0 };
        int32_t len = Seq.size();
        std::vector<uint32_t> ret(len);

        for (int32_t i = 0; i < len; i++)
        {

            const int M = i % K;

            for (int32_t k = M; k < K; ++k)
            {
                pipeline[k] ^= Seq[i];
                pipeline[k] *= 1099511628211;
            }

            //Collect head of pipeline
            ret[i] = pipeline[M];

            for (int32_t k = 0; k < M; ++k)
            {
                pipeline[k] ^= Seq[i];
                pipeline[k] *= 1099511628211;
            }

            pipeline[M] = 2166136261;
        }

        std::partial_sort(ret.begin(), ret.begin() + 200, ret.end());

        auto first = ret.begin();
        auto last = ret.begin() + 200;

        std::vector<uint32_t> newVec(first, last);

        return newVec;

        return ret;
    }

    template<uint32_t K>
    std::vector<uint32_t> generateShinglesSingleHashPipelinePremFast(ContainerType Seq)
    {
        uint32_t pipeline[K] = { 0 };
        uint32_t len = Seq.size();
        std::vector<uint32_t> ret(len);

        for (uint32_t i = 0; i < len; ++i)
        {
            // const int M = i % K;
            #pragma unroll
            for (uint32_t k = 0; k < K; ++k)
            {
                pipeline[k] ^= Seq[i];
                pipeline[k] *= 1099511628211;
            }

            ret[i] = pipeline[0];
            pipeline[0] = 2166136261;
        }

        std::partial_sort(ret.begin(), ret.begin() + 200, ret.end());

        auto first = ret.begin();
        auto last = ret.begin() + 200;

        std::vector<uint32_t> newVec(first, last);

        return newVec;
    }

    double JaccardSingleHash(const std::vector<uint32_t> &seq1, const std::vector<uint32_t> &seq2)
    {
        int len1 = seq1.size();
        int len2 = seq2.size();
        int nintersect = 0;
        int pos1 = 0;
        int pos2 = 0;

        while (pos1 < len1 && pos2 < len2)
        {
            if (seq1[pos1] == seq2[pos2])
            {
                nintersect++;
                pos1++;
                pos2++;
            }
            else if (seq1[pos1] < seq2[pos2])
            {
                pos1++;
            }
            else {
                pos2++;
            }
        }

        int nunion = len1 + len2 - nintersect;
        return nintersect / (double)nunion;
    }*/

};
