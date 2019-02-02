package upmc.ri.struct.instantiation;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.RelatednessCalculator;
import edu.cmu.lti.ws4j.impl.WuPalmer;

public class MultiClassHier extends MultiClass {
	double[][] distances;

	public MultiClassHier(){
		super();
		
		int n = this.dict.size();
		/* Init to 0 */
		this.distances = new double[n][n];
		ILexicalDatabase bd = new NictWordNet();
		RelatednessCalculator calculator = new WuPalmer(bd);

		double min = Double.MAX_VALUE;
		double max = - Double.MAX_VALUE;
		
		for(String class1: enumerateY()) {
			int ychap = this.dict.get(class1);
			
			for(String class2: enumerateY()) {
				int y = this.dict.get(class2);
				
				if (ychap < y) {
					this.distances[ychap][y] = 1 - calculator.calcRelatednessOfWords(class1, class2);
					
					if (this.distances[ychap][y] < min)
						min = this.distances[ychap][y];
					
					if (this.distances[ychap][y] > max)
						max = this.distances[ychap][y];
				}
			}
		}
		
		/* Linear normalization and copy on "the other side" of the array */
		for (int i=1; i < n-1; i++){
			for (int j=i+1; j < n; j++){
				this.distances[i][j] = 1.9 * (this.distances[i][j] - min) / (max - min) + .1;
				this.distances[j][i] = this.distances[i][j];
			}
		}
	}
	
	@Override
	public double delta(String y1, String y2) {
		return this.distances[this.dict.get(y1)][this.dict.get(y2)];
	}
}
