package upmc.ri.struct.model;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;
import upmc.ri.utils.VectorOperations;

public class LinearStructModel_Ex<X,Y> extends LinearStructModel<X,Y> {
	public LinearStructModel_Ex(IStructInstantiation<X,Y> instance, int dimpsi) {
		super(dimpsi);
		this.instance = instance;
	}

	@Override
	public Y predict(STrainingSample<X, Y> ts) {
		X xi = ts.input;
		Y maxY = null;
		double max = -Double.MAX_VALUE;
		
		for (Y y : this.instance.enumerateY()) {
			
			double psi = VectorOperations.dot(this.getParameters(), this.instance.psi(xi, y));
			
			if (max < psi) {
				max = psi;
				maxY = y;
			}
		}
		return maxY;
	}
	
	@Override
	public Y lai(STrainingSample<X, Y> ts) {		
		X xi = ts.input;
		Y yi = ts.output;
		Y maxY = null;
		double maxLai = -Double.MAX_VALUE;
		
		for (Y y : this.instance.enumerateY()) {
			double delta = this.instance.delta(y, yi);
			double psi = VectorOperations.dot(this.getParameters(), this.instance.psi(xi, y));
			
			double currLai = delta + psi;
			if (maxLai < currLai) {
				maxLai = currLai;
				maxY = y;
			}
		}
		return maxY;
	}
}
