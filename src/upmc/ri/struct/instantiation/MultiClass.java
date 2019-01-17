package upmc.ri.struct.instantiation;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.MatrixVisualization;

import upmc.ri.io.ImageNetParser;

public class MultiClass implements IStructInstantiation<double[], String> {
	protected Map<String,Integer> dict;
	public MultiClass(){
		this.dict = new HashMap();
		Integer count = 0;
		for(String classe:enumerateY()) {
			this.dict.put(classe, count);
			count ++;
		}
	}
	
	public void confusionMatrix(List<String> predictions, List<String> gt) {
		double[] a = new double[this.dict.size() * this.dict.size()];
		DenseMatrix64F matrice = new DenseMatrix64F(this.dict.size(), this.dict.size());
		for(int i=0; i< predictions.size() ; i++){
			int ŷId = this.dict.get(predictions.get(i));
			int yId = this.dict.get(gt.get(i));
			double old_value = matrice.get(ŷId, yId);
			matrice.set(ŷId, yId, old_value + 1);
		}
		System.out.println(matrice.getData());
		MatrixVisualization.show(matrice, "Matrice de confusion de la base de test <3 <3");
	}
	
	@Override
	public double[] psi(double[] x, String y) {
		Integer id = this.dict.get(y);
		double[] vect = new double[x.length * this.dict.size()];
		for(int i = 0 ; i < x.length ; i++) {
			vect[i + id*x.length] = x[i];
		}
		return vect;
	}

	@Override
	public double delta(String y1, String y2) {
		if (y1.equals(y2)) {
			return 0;
		}
		return 1;
	}

	@Override
	public Set<String> enumerateY() {
		return ImageNetParser.classesImageNet();
	}

}
