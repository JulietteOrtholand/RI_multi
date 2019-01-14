package upmc.ri.struct.training;

import java.util.List;
import java.util.Random;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;
import upmc.ri.struct.model.IStructModel;
import upmc.ri.utils.VectorOperations;

public class SGDTrainer<X,Y> implements ITrainer<X,Y>{
	
	private Evaluator<X, Y> evaluator;
	private int max_iter;
	private double lambda;
	private double eps;

	public SGDTrainer(Evaluator<X, Y> evaluator, double eps, double lambda, int max_iter) {
		this.evaluator = evaluator;
		this.max_iter = max_iter;
		this.lambda = lambda;
		this.eps = eps;
	}
	
	@Override
	public void train(List<STrainingSample<X, Y>> lts, IStructModel<X, Y> model) {
		Random rand = new Random();

		/* w <- 0 */
		double[] params = model.getParameters();
		for (int i = 0; i < params.length; i++) {
			params[i] = 0.0;
		}
		
		IStructInstantiation<X, Y> instance = model.instantiation();
		
		for (int t = 0; t < this.max_iter; t++) {
			for (int i = 0; i < lts.size(); i++) {
				/* random selection */
				STrainingSample<X, Y> ts = lts.get(rand.nextInt(lts.size()));
				X xi = ts.input;
				Y yi = ts.output;
				
				/* loss-augmented inference */
				Y ŷ = model.lai(ts);
				
				double[] g = VectorOperations.subtract(
						instance.psi(xi, ŷ), 
						instance.psi(xi, yi)
						);
				
				/* update */
				for (int j = 0; j < params.length; j++) {
					params[j] = params[j] - this.eps * (this.lambda * params[j] + g[j]);
				}
			}
			this.evaluator.evaluate();
			System.out.println("epoch : " + t);
			System.out.println("global loss : " + this.convex_loss(lts, model));
			System.out.println("err train : " + this.evaluator.getErr_train());
			System.out.println("err test : " + this.evaluator.getErr_test());
		}
	}
	
	private double convex_loss(List<STrainingSample<X, Y>> lts , IStructModel<X,Y> model) {
		double loss = 0;
		double[] params = model.getParameters();
		IStructInstantiation<X, Y> instantiation = model.instantiation();
		
		for (int i = 0; i < lts.size(); i++) {
			
			STrainingSample<X, Y> ts = lts.get(i);
			
			X xi = ts.input;
			Y yi = ts.output;
			double max = -Double.MAX_VALUE; //check if same as -999999.99;
			
			for (Y y : instantiation.enumerateY()) {
				max = Math.max( max, instantiation.delta(yi, y) + VectorOperations.dot(instantiation.psi(xi, y), params) );
			}
			loss += max - VectorOperations.dot(instantiation.psi(xi, yi), params);
		}
		loss /= lts.size();
		loss += this.lambda / 2 * VectorOperations.norm2(params);
		return loss;
	}
}
