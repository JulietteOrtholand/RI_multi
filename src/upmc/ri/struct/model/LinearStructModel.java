package upmc.ri.struct.model;

import upmc.ri.struct.instantiation.IStructInstantiation;

public abstract class LinearStructModel<X,Y> implements IStructModel<X,Y> {

	protected IStructInstantiation<X,Y> structInst;
	protected double[] parameters;
	protected int dimpsi;
	
	public double[] getParameters() {
		return this.parameters;
	}
	
	public LinearStructModel(int dimpsi){
		this.parameters = new double[2];
		this.parameters[1] = dimpsi;
	}
	
}
