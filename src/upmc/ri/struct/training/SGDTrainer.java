package upmc.ri.struct.training;

import java.util.List;

import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;
import upmc.ri.struct.model.IStructModel;

public class SGDTrainer<X,Y> implements ITrainer<X,Y>{
	
	private Evaluator<X, Y> eval;
	@Override
	public void train(List<STrainingSample<X, Y>> lts, IStructModel<X, Y> model) {
		// TODO Auto-generated method stub
		double[] params = model.getParameters();
		double[] w = new double[(int) params[1]];
		IStructInstantiation<X,Y> instant = model.instantiation();
		for(int t = 0; t< params[0]; t++ ) {
			//choisir un couple de maniere aleatoire
			for(STrainingSample<X, Y> couple : lts) {
				Y yChap = model.predict(couple);
				double[] g = instant.psi(couple.input, yChap) - instant.psi(couple.input, couple.output);
				model.setW(model.getW() - this.step *(this.reg*model.getW() + g));
			}
		}
	}
	
	public void convex_loss() {
	
	}
}
