package upmc.ri.struct.model;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;

public class LinearStructModel_Ex<X,Y> extends LinearStructModel<X,Y> {
	protected IStructInstantiation <X,Y> instance;
	protected double[] w = new double[(int) this.getParameters()[1]];
	
	public LinearStructModel_Ex(int dimpsi) {
		super(dimpsi);
		this.instance= this.instantiation();
		// TODO Auto-generated constructor stub
	}

	@Override
	public Y predict(STrainingSample<X, Y> ts) {
		return this.lai(ts);
	}
	
	@Override
	public Y lai(STrainingSample<X, Y> couple) {
		int dimpsi = (int) this.getParameters()[1];
		
		double maxLai = 0;
		int maxIdx = -1;
		//Utiliser enumerateY ...............
		for(int j; j < dimpsi; j++) {
			double pv = 0;
			double[] psi = this.instance.psi(couple.input, j);
			double delta = this.instance.delta(couple.output, j);
			
			for(int i=0; i < psi.length; i++) {
				pv += w[i]*psi[i];
			}
			
			double currLai = pv + delta;
			if (maxLai < currLai || maxIdx == -1) {
				maxLai = currLai;
				maxIdx = j;
			}
		}
		return maxIdx;
	}

	@Override
	public IStructInstantiation<X, Y> instantiation() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public double[] getW() {
		return this.w;
	}
	
	public void setW(double[] w) {
		this.w = w;
	}

}
