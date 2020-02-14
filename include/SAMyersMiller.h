template <typename ContainerType, typename Ty = typename ContainerType::value_type, Ty Blank = Ty(0), typename MatchFnTy = std::function<bool(Ty, Ty)>>
class MyersMillerSA : public SequenceAligner<ContainerType, Ty, Blank, MatchFnTy>
{
  private:
	size_t vectorSize;		//M
	size_t smallVectorSize; //N
	size_t MatchesRows;
	size_t MatchesCols;

	using BaseType = SequenceAligner<ContainerType, Ty, Blank, MatchFnTy>;

	ScoringSystem &Scoring = BaseType::getScoring();
	const ScoreSystemType Match = Scoring.getMatchProfit();
	const bool AllowMismatch = Scoring.getAllowMismatch();
	const ScoreSystemType Mismatch = AllowMismatch
										 ? Scoring.getMismatchPenalty()
										 : std::numeric_limits<ScoreSystemType>::min();
	const float GapOpen = (float)Scoring.getGapOpenPenalty();
	const float GapExtend = (float)Scoring.getGapExtendPenalty();

	//Save all the matches to memory
	bool *cacheAllMatches(ContainerType &Seq1, ContainerType &Seq2)
	{
		bool *Matches = nullptr;
		if (BaseType::getMatchOperation() == nullptr)
		{
			Matches = nullptr;
			return nullptr;
		}
		const size_t SizeSeq1 = Seq1.size();
		const size_t SizeSeq2 = Seq2.size();

		MatchesRows = SizeSeq1;
		MatchesCols = SizeSeq2;
		Matches = new bool[SizeSeq1 * SizeSeq2];
		for (unsigned i = 0; i < SizeSeq1; i++)
			for (unsigned j = 0; j < SizeSeq2; j++)
				Matches[i * SizeSeq2 + j] = BaseType::match(Seq1[i], Seq2[j]);

		return Matches;
	}

	//Compute the Score Vectors
	std::pair<float *, float *> computeScoreVectors(ContainerType &Seq1, ContainerType &Seq2, float tbe)
	{
		//initialize Vector and scoring elements
		const size_t SizeSeq1 = Seq1.size();
		const size_t SizeSeq2 = Seq2.size();

		bool *Matches = cacheAllMatches(Seq1, Seq2);

		const size_t NumRows = SizeSeq1 + 1;
		const size_t NumCols = SizeSeq2 + 1;
		vectorSize = NumRows > NumCols ? SizeSeq1 + 1 : SizeSeq2 + 1;
		smallVectorSize = NumRows > NumCols ? SizeSeq2 + 1 : SizeSeq1 + 1;

		float *CC = new float[smallVectorSize];
		float *DD = new float[smallVectorSize];

		float e, c, s, t = 0;

		//Set up initial values of the vectors
		//For Myers and Miller we know that CC and DD should be initialised as
		//Gap opening and extending

		//Set first value of CC
		CC[0] = 0;

		int g = GapOpen;
		float h = GapExtend;

		t = g;

		//Set initial values of CC and DD
		for (unsigned int i = 1; i < smallVectorSize; i++)
		{
			t += GapExtend;
			CC[i] = t;
			DD[i] = t + GapOpen;
		}

		t = g;
		//No if statements within these loops for efficiency
		//TODO CHECK IF NEEDS ALTERED
		ScoreSystemType MaxScore = std::numeric_limits<ScoreSystemType>::min();

		if (Matches)
		{
			if (AllowMismatch)
			{
				for (unsigned int i = 1; i < vectorSize; i++)
				{
					s = CC[0];
					t = t + h;
					c = t;
					CC[0] = c;
					e = t + g;

					for (unsigned int j = 1; j < smallVectorSize; j++)
					{
						e = std::max(e, c + g) + h;
						DD[j] = std::max(DD[j], CC[j] + g) + h;
						ScoreSystemType Similarity = Matches[(i - 1) * MatchesCols + j - 1] ? Match : Mismatch;
						c = std::max({DD[j], e, s + Similarity});
						s = CC[j];
						CC[j] = c;
					}
				}
			}
			else
			{ //No mismatch allowed

				for (unsigned int i = 1; i < vectorSize; i++)
				{
					s = CC[0];
					t = t + h;
					c = t;
					CC[0] = c;
					e = t + g;

					for (unsigned int j = 1; j < smallVectorSize; j++)
					{
						e = std::max(e, c + g) + h;
						DD[j] = std::max(DD[j], CC[j] + g) + h;
						float Similarity = Matches[(i - 1) * MatchesCols + j - 1] ? Match : Mismatch;
						if (Matches[(i - 1) * MatchesCols + j - 1])
						{
							c = std::max({DD[j], e, s + Similarity});
						}
						else
						{
							c = std::max({DD[j], e, Similarity});
						}
						s = CC[j];
						CC[j] = c;
					}
				}
			}
		}
		else
		{ //No matches matrix
			if (AllowMismatch)
			{
				for (unsigned int i = 1; i < vectorSize; i++)
				{
					s = CC[0];
					t = t + h;
					c = t;
					CC[0] = c;
					e = t + g;

					for (unsigned int j = 1; j < smallVectorSize; j++)
					{
						e = std::max(e, c + g) + h;
						DD[j] = std::max(DD[j], CC[j] + g) + h;
						ScoreSystemType Similarity = (Seq1[i - 1] == Seq2[j - 1]) ? Match : Mismatch;
						c = std::max({DD[j], e, s + Similarity});
						s = CC[j];
						CC[j] = c;
					}
				}
			}
			else
			{ //No mismatch allowed
				for (unsigned int i = 1; i < vectorSize; i++)
				{
					s = CC[0];
					t = t + h;
					c = t;
					CC[0] = c;
					e = t + g;

					for (unsigned int j = 1; j < smallVectorSize; j++)
					{
						e = std::max(e, c + g) + h;
						DD[j] = std::max(DD[j], CC[j] + g) + h;
						float Similarity = (Seq1[i - 1] == Seq2[j - 1]) ? Match : Mismatch;
						if (Seq1[i - 1] == Seq2[j - 1])
						{
							c = std::max({DD[j], e, s + Similarity});
						}
						else
						{
							c = std::max({DD[j], e, 0 + Similarity});
						}
						s = CC[j];
						CC[j] = c;
					}
				}
			}
		}

		auto pair = std::make_pair(CC, DD);
		return pair;
	}

	//Build the resulting aligned sequence
	void buildResultRec(ContainerType &Seq1, ContainerType &Seq2, AlignedSequence<Ty, Blank> &Result, int M, int N, float tb, float te)
	{

		//Because of recursive nature and sequences being shortened
		//The matches matrix must be updated as it still maintains
		//The full length sequences matches data

		bool *Matches = nullptr;

		Matches = cacheAllMatches(Seq1, Seq2);

		auto &Data = Result.Data;

		if (N == 0)
		{
			if (M > 0)
			{
				for (auto Char : Seq1)
				{
					Data.push_back(
						typename BaseType::EntryType(Char, Blank, false));
				}
			}
		}
		else if (M == 0)
		{

			for (auto Char : Seq2)
			{
				Data.push_back(
					typename BaseType::EntryType(Blank, Char, false));
			}
		}
		else if (M == 1)
		{
			//gap(k) = 2 + 0.5k
			float max = std::max(tb, te) + GapExtend + (GapExtend * N);
			float maxLoop = -100000;
			int index = 0;

			for (int j = 1; j < N + 1; j++)
			{

				ScoreSystemType Similarity;
				/*if (Matches != false) {
						Similarity = Matches[MatchesCols + j - 1] ? Match : Mismatch;
					}
					else {
						Similarity = Seq1[0] == Seq2[j] ? Match : Mismatch;
					}*/

				Similarity = Seq1[0] == Seq2[j - 1] ? Match : Mismatch;

				float maxTemp = std::max(GapOpen + (GapExtend * (j - 1)) + Similarity + GapOpen + (GapExtend * (N - j)), max);

				if (maxTemp > maxLoop)
				{
					maxLoop = maxTemp;
					index = j;
				}
			}

			for (int i = 1; i < N + 1; i++)
			{
				if (i == index)
				{

					bool isValidMatch = Seq1[0] == Seq2[i - 1];

					Data.push_back(
						typename BaseType::EntryType(Seq1[0], Seq2[i - 1], isValidMatch));
				}
				else
				{
					Data.push_back(
						typename BaseType::EntryType(Blank, Seq2[i - 1], false));
				}
			}
		}
		else
		{

			int mid = (int)(M / 2);

			//Find maximum path scores from (i1, j1)
			float *CC = new float[N + 1];
			float *DD = new float[N + 1];
			float s = 0;
			float c = 0;

			CC[0] = 0;

			int g = GapOpen;
			float h = GapExtend;

			float t = 0;
			float e = 0;

			t = g;

			//Set initial values of CC and DD
			for (unsigned int i = 1; i < N + 1; i++)
			{
				t += GapExtend;
				CC[i] = t;
				DD[i] = t + GapOpen;
			}

			t = tb;

			for (unsigned int i = 1; i < mid + 1; i++)
			{
				s = CC[0];
				t = t + h;
				c = t;
				CC[0] = c;
				e = t + g;

				for (unsigned int j = 1; j < N + 1; j++)
				{
					e = std::max(e, c + g) + h;
					DD[j] = std::max(DD[j], CC[j] + g) + h;
					ScoreSystemType Similarity = Matches[(i - 1) * MatchesCols + j - 1] ? Match : Mismatch;
					c = std::max({DD[j], e, s + Similarity});
					s = CC[j];
					CC[j] = c;
				}
			}

			DD[0] = CC[0];

			//Find maximum path scores to (i2, j2)

			float *RR = new float[N + 1];
			float *SS = new float[N + 1];
			s = 0;
			c = 0;

			RR[N] = 0;

			g = GapOpen;
			h = GapExtend;

			t = 0;
			e = 0;

			t = g;

			//Set initial values of RR and SS
			for (int i = N - 1; i > -1; i--)
			{
				t += GapExtend;
				RR[i] = t;
				SS[i] = t + GapOpen;
			}

			t = te;

			for (int i = M - 2; i > mid - 2; i--)
			{
				s = RR[N];
				t = t + h;
				c = t;
				RR[N] = c;
				e = t + g;

				for (int j = N - 1; j > -1; j--)
				{
					e = std::max(e, c + g) + h;
					SS[j] = std::max(SS[j], RR[j] + g) + h;
					ScoreSystemType Similarity = Matches[(i + 1) * MatchesCols + j] ? Match : Mismatch;
					c = std::max({SS[j], e, s + Similarity});
					s = RR[j];
					RR[j] = c;
				}
			}

			SS[N] = RR[N];

			//Find where maximum score path crosses row mid
			int index = 0;
			float max = -10000000;
			bool type = false; //False = type 1 midpoint
							   //True = type 2
			for (int j = 0; j < N + 1; j++)
			{
				float temp = 0;

				temp = std::max(CC[j] + RR[j], DD[j] + SS[j] - g);

				if (temp > max)
				{
					max = temp;
					index = j;

					if (CC[j] + RR[j] > DD[j] + SS[j] - g)
					{
						type = false;
					}
					else
					{
						type = true;
					}
				}
			}

			if (CC)
				delete[] CC;
			CC = nullptr;
			if (DD)
				delete[] DD;
			DD = nullptr;
			if (SS)
				delete[] SS;
			SS = nullptr;
			if (RR)
				delete[] RR;
			RR = nullptr;
			if (Matches)
				delete[] Matches;
			Matches = nullptr;

			if (!type)
			{ //If a type 1 midpoint
				//Recurse on the newfound indices
				std::string newSeq1 = Seq1.substr(0, mid);
				std::string newSeq2 = Seq2.substr(0, index);
				std::string newSeq1Temp = Seq1.substr(mid, Seq1.size());
				std::string newSeq2Temp = Seq2.substr(index, Seq2.size());

				buildResultRec(newSeq1, newSeq2, Result, mid, index, tb, g);
				buildResultRec(newSeq1Temp, newSeq2Temp, Result, M - mid, N - index, g, te);
			}
			else
			{ //If a type 2 midpoint
				//Recurse on the newfound indices
				std::string newSeq1 = Seq1.substr(0, mid - 1);
				std::string newSeq2 = Seq2.substr(0, index);
				std::string newSeq1Temp = Seq1.substr(mid + 1, Seq1.size());
				std::string newSeq2Temp = Seq2.substr(index, Seq2.size());

				buildResultRec(newSeq1, newSeq2, Result, mid - 1, index, tb, 0);
				//Write delete ai* ai*+1
				Data.push_back(
					typename BaseType::EntryType(Seq1[mid - 1], Blank, false));
				Data.push_back(
					typename BaseType::EntryType(Seq1[mid], Blank, false));
				buildResultRec(newSeq1Temp, newSeq2Temp, Result, M - mid - 1, N - index, 0, te);
			}
		}
	}

	void clearAll()
	{
		//if (Matches) delete[] Matches;
		//Matches = nullptr;
	}

  public:
	static ScoringSystem getDefaultScoring()
	{
		return ScoringSystem(-1, 2, -1);
	}

	MyersMillerSA() : BaseType(getDefaultScoring(), nullptr) {}

	MyersMillerSA(ScoringSystem Scoring, MatchFnTy Match = nullptr)
		: BaseType(Scoring, Match) {}

	virtual AlignedSequence<Ty, Blank> getAlignment(ContainerType &Seq1, ContainerType &Seq2)
	{
		AlignedSequence<Ty, Blank> Result;
		//cacheAllMatches(Seq1,Seq2);
		//computeScoreVectors(Seq1,Seq2, 0);
		buildResultRec(Seq1, Seq2, Result, Seq1.size(), Seq2.size(), GapOpen, GapOpen);
		clearAll();
		return Result;
	}
};
