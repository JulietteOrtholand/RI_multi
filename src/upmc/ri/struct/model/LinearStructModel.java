package upmc.ri.struct.model;

import upmc.ri.struct.instantiation.IStructInstantiation;

public abstract class LinearStructModel<X,Y> implements IStructModel<X,Y> {

	protected IStructInstantiation<X,Y> instance;
	protected double[] parameters;
	protected int dimpsi;

	public LinearStructModel(int dimpsi){
		this.parameters = new double[dimpsi];
	}
	
	@Override
	public double[] getParameters() {
		return this.parameters;
	}
	
	@Override
	public IStructInstantiation<X, Y> instantiation() {
		return this.instance;
	}

	
	@Override
	public void setInstantiation(IStructInstantiation <X,Y> instance) {
		this.instance = instance;
	}
}
