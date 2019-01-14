package upmc.ri.struct.model;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;

public interface IStructModel<X,Y> {
	public Y predict(STrainingSample<X,Y> ts);
	public Y lai(STrainingSample<X,Y> ts);
	public IStructInstantiation <X,Y> instantiation();
	void setInstantiation(IStructInstantiation<X, Y> instance);
	public double[] getParameters();
	// [dim]
}
