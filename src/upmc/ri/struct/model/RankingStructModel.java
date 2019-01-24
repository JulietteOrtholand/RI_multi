package upmc.ri.struct.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.utils.VectorOperations;

public class RankingStructModel extends LinearStructModel<List<double[]>, RankingOutput>{

	public RankingStructModel(IStructInstantiation<List<double[]>, RankingOutput> instance, int dimpsi) {
		super(dimpsi);
		this.instance = instance;
	}

	@Override
	public RankingOutput predict(STrainingSample<List<double[]>, RankingOutput> ts) {

		List<double[]> x = ts.input;
		List<Result> results = new ArrayList<Result>();
		for (int i=0; i < x.size(); i++) {
			results.add(new Result(VectorOperations.dot(this.parameters, x.get(i)), i));
		}
		
		// Sorting using the collection compare to function
		Collections.sort(results, Collections.reverseOrder());
		
		List<Integer> ranking = new ArrayList<Integer>();
		for (int i=0; i < results.size(); i++) {
			ranking.add(results.get(i).id);
		}
		return new RankingOutput(ts.output.getNbPlus(), ranking, ts.output.getLabelsGT());
	}

	public RankingOutput lai(STrainingSample<List<double[]>, RankingOutput> ts) {
		return RankingFunctions.loss_augmented_inference(ts, this.parameters);
	}
	

	// https://stackoverflow.com/questions/21626439/how-to-implement-the-java-comparable-interface
	class Result implements Comparable<Result> {
		
		public double result;
		public int id;
		
		public Result (double result, int id) {
			this.result = result;
			this.id = id;
		}

		@Override
		public int compareTo(Result o) {
			if (this.result > o.result)
				return 1;
			else if (this.result < o.result)
				return -1;
			return 0;
		}
	}

}
