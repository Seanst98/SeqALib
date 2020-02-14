#include <queue>
template <typename ContainerType, typename Ty = typename ContainerType::value_type, Ty Blank = Ty(0), typename MatchFnTy = std::function<bool(Ty, Ty)>>
class FOGSAASA : public SequenceAligner<ContainerType, Ty, Blank, MatchFnTy>
{
  private:
	bool *Matches;
	size_t MatchesRows;
	size_t MatchesCols;
	int M;
	int N;

	ScoringSystem &Scoring = BaseType::getScoring();
	const ScoreSystemType Match = Scoring.getMatchProfit();
	const bool AllowMismatch = Scoring.getAllowMismatch();
	const ScoreSystemType Mismatch = AllowMismatch
										 ? Scoring.getMismatchPenalty()
										 : std::numeric_limits<ScoreSystemType>::min();
	const ScoreSystemType GapOpen = Scoring.getGapOpenPenalty();
	const ScoreSystemType GapExtend = Scoring.getGapExtendPenalty();

	using BaseType = SequenceAligner<ContainerType, Ty, Blank, MatchFnTy>;

	class Node
	{
	  public:
		Node() {}

		int P1 = 0;
		int P2 = 0;
		int presentScore = std::numeric_limits<int>::min();
		int Tmin = std::numeric_limits<int>::min();
		int Tmax = std::numeric_limits<int>::min();
		AlignedSequence<Ty, Blank> Seq;

		Node *operator=(const Node b)
		{
			this->Tmax = b.Tmax;
			this->Tmin = b.Tmin;
			this->P1 = b.P1;
			this->P2 = b.P2;
			this->presentScore = b.presentScore;
			this->Seq.Data = b.Seq.Data;
			return this;
		}

		bool operator<(Node &b) const
		{
			return Tmax > b.Tmax;
		}

		/*//Use Tmax to sort priority queue
		//On the event where Tmax are equal
		//Use Tmin
		bool operator>(const Node b) const
		{
			std::cout << "We usin this one" << std::endl;
			if (Tmax == b.Tmax)
				return Tmin < b.Tmin;
			return Tmax < b.Tmax;
		}*/

		bool operator>(const Node b) const
		{
			return Tmin < b.Tmin;
		}

		void calculateScores(int newP1, int newP2, int M, int N, AlignedSequence<Ty, Blank> &Result, ScoringSystem &scoreSystem)
		{

			const ScoreSystemType Gap = scoreSystem.getGapPenalty();
			const ScoreSystemType Match = scoreSystem.getMatchProfit();
			const bool AllowMismatch = scoreSystem.getAllowMismatch();
			const ScoreSystemType Mismatch = AllowMismatch ? scoreSystem.getMismatchPenalty() : std::numeric_limits<ScoreSystemType>::min();

			P1 = newP1;
			P2 = newP2;

			//Calculate Future scores (Fmin, Fmax)
			int Fmax = 0;
			int Fmin = 0;
			int x1 = (N - P2);
			int x2 = (M - P1);
			if (x2 < x1)
			{
				Fmin = (x2 * Mismatch) + (Gap * (x1 - x2));
				Fmax = (x2 * Match) + (Gap * (x1 - x2));
			}
			else
			{
				Fmin = (x1 * Mismatch) + (Gap * (x2 - x1));
				Fmax = (x1 * Match) + (Gap * (x2 - x1));
			}

			//Add Present score to the Future score
			//obtaining the fitness score
			Tmin = presentScore + Fmin;
			Tmax = presentScore + Fmax;

			Seq.Data = Result.Data;
		}
	};

	class HashPriorityQueue
	{
		int numBuckets;
		int min;
		int max;

		std::vector<std::priority_queue<Node, std::vector<Node>, std::greater<Node>>> *table;

	  public:
		HashPriorityQueue(int min, int max)
		{
			this->numBuckets = max - min;
			this->min = min;
			this->max = max;
			table = new std::vector<std::priority_queue<Node, std::vector<Node>, std::greater<Node>>>;
			table->resize(numBuckets);
		}

		int insertItem(int key, Node node, int maxPointer)
		{
			int index = hashFunction(key);
			table->at(index).push(node);

			//First entry so maxPointer is set to the one being inserted
			if (maxPointer == min - 1)
			{
				return node.Tmax;
			}

			//If the one we are inserting is better than what we are currently
			//Pointing towards then return
			//A new pointer to that position in the hash
			if (node.Tmax > maxPointer)
			{
				maxPointer = node.Tmax;
			}

			return maxPointer;
		}

		//Get top from the hashed queue and pop it too
		Node getTop(int key)
		{
			int index = hashFunction(key);
			if (table->at(index).size() == 0)
			{
				Node a;
				return a;
			}
			return table->at(index).top();
		}

		//Delete top item and return a new maxPointer
		int deleteTop(int key)
		{
			int index = hashFunction(key);
			table->at(index).pop();

			int maxPointer = key;

			int i = index;
			while (table->at(i).empty())
			{
				i++;
				maxPointer--;
			}

			return maxPointer;
		}

		int hashFunction(int key)
		{
			return max - key;
		}
	};

	Node *c;

	//Save all the matches to memory
	void cacheAllMatches(ContainerType &Seq1, ContainerType &Seq2)
	{
		if (BaseType::getMatchOperation() == nullptr)
		{
			Matches = nullptr;
			return;
		}
		const size_t SizeSeq1 = Seq1.size();
		const size_t SizeSeq2 = Seq2.size();

		MatchesRows, M = SizeSeq1;
		MatchesCols, N = SizeSeq2;
		Matches = new bool[SizeSeq1 * SizeSeq2];
		c = new Node[(SizeSeq1 + 1) * (SizeSeq2 + 1) + 2];
		for (unsigned i = 0; i < SizeSeq1; i++)
			for (unsigned j = 0; j < SizeSeq2; j++)
				Matches[i * SizeSeq2 + j] = BaseType::match(Seq1[i], Seq2[j]);
	}

	//Build the resulting aligned sequence
	void buildAlignment(ContainerType &Seq1, ContainerType &Seq2, AlignedSequence<Ty, Blank> &Result)
	{

		ScoringSystem &Scoring = BaseType::getScoring();
		const ScoreSystemType Gap = Scoring.getGapPenalty();
		const ScoreSystemType Match = Scoring.getMatchProfit();
		const bool AllowMismatch = Scoring.getAllowMismatch();
		const ScoreSystemType Mismatch = AllowMismatch ? Scoring.getMismatchPenalty() : std::numeric_limits<ScoreSystemType>::min();

		int P1 = 0;
		int P2 = 0;
		int optimal = std::numeric_limits<int>::min();

		int newTmax = 0;
		int maxTmax = 0;
		Node currentNode;
		currentNode.presentScore = 0;
		Node optimalNode;

		int Fmax = 0;
		int Fmin = 0;
		int x1 = (N - P2);
		int x2 = (M - P1);
		if (x2 < x1)
		{
			Fmin = (x2 * Mismatch) + (Gap * (x1 - x2));
			Fmax = (x2 * Match) + (Gap * (x1 - x2));
		}
		else
		{
			Fmin = (x1 * Mismatch) + (Gap * (x2 - x1));
			Fmax = (x1 * Match) + (Gap * (x2 - x1));
		}

		HashPriorityQueue *hpqueue = new HashPriorityQueue(Fmin, Fmax);
		int maxPointer = Fmin - 1; //Points towards top of hashed priority queue
								   //Initially points towards nothing

		if (M != 0 && N != 0)
		{
			do
			{
				while (P1 <= (M - 1) || P2 <= (N - 1))
				{

					if (currentNode.presentScore > c[P1 * N + P2].presentScore)
					{
						c[P1 * N + P2] = currentNode;
					}

					//Select the best child from the remaining children according to the Tmax
					//Develop sequence by adding either Match, Mismatch, or Gap
					//From each developed sequence, calculate fitness scores
					int x = P1;
					int y = P2;
					bool IsValidMatch = false;
					if (Matches)
					{
						IsValidMatch = Matches[(P1)*N + P2];
					}
					else
					{
						IsValidMatch = (Seq1[P1] == Seq2[P2]);
					}

					//Match/Mismatch
					AlignedSequence<Ty, Blank> MMSeq = currentNode.Seq;
					MMSeq.Data.push_back(
						typename BaseType::EntryType(Seq1[P1], Seq2[P2], IsValidMatch));

					//Gap added to first sequence
					AlignedSequence<Ty, Blank> _GSeq = currentNode.Seq;
					_GSeq.Data.push_back(
						typename BaseType::EntryType(Blank, Seq2[P2], false));

					//Gap added to second sequence
					AlignedSequence<Ty, Blank> G_Seq = currentNode.Seq;
					G_Seq.Data.push_back(
						typename BaseType::EntryType(Seq1[P1], Blank, false));

					Node MMNode;
					Node _GNode;
					Node G_Node;
					Node childNode;

					int Similarity = IsValidMatch ? Match : Mismatch;
					MMNode.presentScore = currentNode.presentScore + Similarity;
					_GNode.presentScore = currentNode.presentScore + Gap;
					G_Node.presentScore = currentNode.presentScore + Gap;

					MMNode.calculateScores(P1 + 1, P2 + 1, M, N, MMSeq, Scoring);
					_GNode.calculateScores(P1, P2 + 1, M, N, _GSeq, Scoring);
					G_Node.calculateScores(P1 + 1, P2, M, N, G_Seq, Scoring);

					if (P1 > (M - 1))
					{
						MMNode.Tmax = std::numeric_limits<int>::min();
						G_Node.Tmax = std::numeric_limits<int>::min();
					}
					if (P2 > (N - 1))
					{
						MMNode.Tmax = std::numeric_limits<int>::min();
						_GNode.Tmax = std::numeric_limits<int>::min();
					}

					if (MMNode.Tmax >= std::max(_GNode.Tmax, G_Node.Tmax))
					{
						childNode = MMNode;
						P1++;
						P2++;
						maxPointer = hpqueue->insertItem(_GNode.Tmax, _GNode, maxPointer);
						maxPointer = hpqueue->insertItem(G_Node.Tmax, G_Node, maxPointer);
					}
					else if (G_Node.Tmax > std::max(MMNode.Tmax, _GNode.Tmax))
					{
						childNode = G_Node;
						P1++;
						maxPointer = hpqueue->insertItem(_GNode.Tmax, _GNode, maxPointer);
						maxPointer = hpqueue->insertItem(MMNode.Tmax, MMNode, maxPointer);
					}
					else
					{
						childNode = _GNode;
						P2++;
						maxPointer = hpqueue->insertItem(MMNode.Tmax, MMNode, maxPointer);
						maxPointer = hpqueue->insertItem(G_Node.Tmax, G_Node, maxPointer);
					}

					if (childNode.presentScore <= c[P1 * N + P2].presentScore)
					{
						//Prune the current branch
						childNode = hpqueue->getTop(maxPointer);
						maxPointer = hpqueue->deleteTop(maxPointer);
						P1 = childNode.P1;
						P2 = childNode.P2;
					}
					else
					{

						c[P1 * N + P2] = childNode;
						if (childNode.Tmax <= optimal)
						{
							//Prune the current branch
							childNode = hpqueue->getTop(maxPointer);
							maxPointer = hpqueue->deleteTop(maxPointer);
							P1 = childNode.P1;
							P2 = childNode.P2;
						}
					}

					currentNode = childNode;
				}

				if (c[P1 * N + P2].Tmax >= optimal)
				{
					optimal = c[P1 * N + P2].Tmax;
					currentNode = c[P1 * N + P2];
					P1 = currentNode.P1;
					P2 = currentNode.P2;
					optimalNode = currentNode;
				}

				currentNode = hpqueue->getTop(maxPointer);
				maxPointer = hpqueue->deleteTop(maxPointer);
				P1 = currentNode.P1;
				P2 = currentNode.P2;
				newTmax = currentNode.Tmax;

				//If top most node has Tmax so less than 30% similarity then
				//end the process and report approximate score
				int maxScore = 0;
				int minScore = 0;
				if (M > N)
				{
					maxScore = (M - N) * Gap + (N * Match);
					minScore = (M - N) * Gap + (N * Mismatch);
				}
				else
				{
					maxScore = (N - M) * Gap + (M * Match);
					minScore = (N - M) * Gap + (M * Mismatch);
				}

				int range = maxScore - minScore;
				float similarity = (float)(newTmax + std::abs(minScore)) / range;
				if (similarity < 0.3)
				{
					Result.Data = optimalNode.Seq.Data;
					return;
				}

			} while (optimal < newTmax);
		}

		Result.Data = optimalNode.Seq.Data;
	}

	void clearAll()
	{
		if (Matches)
			delete[] Matches;
		Matches = nullptr;

		if (c)
			delete[] c;
		c = nullptr;
	}

  public:
	static ScoringSystem getDefaultScoring()
	{
		return ScoringSystem(-1, 2, -1);
	}

	FOGSAASA() : BaseType(getDefaultScoring(), nullptr), Matches(nullptr) {}

	FOGSAASA(ScoringSystem Scoring, MatchFnTy Match = nullptr)
		: BaseType(Scoring, Match), Matches(nullptr) {}

	virtual AlignedSequence<Ty, Blank> getAlignment(ContainerType &Seq1, ContainerType &Seq2)
	{
		AlignedSequence<Ty, Blank> Result;
		cacheAllMatches(Seq1, Seq2);
		buildAlignment(Seq1, Seq2, Result);
		clearAll();
		return Result;
	}
};
