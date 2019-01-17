package upmc.ri.struct.instantiation;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.RelatednessCalculator;
import edu.cmu.lti.ws4j.impl.WuPalmer;

public class MultiClassHier extends MultiClass {
	double[][] distances;

	public MultiClassHier(){
		super();
		
		this.distances = new double[this.dict.size()][this.dict.size()];
		ILexicalDatabase bd = new NictWordNet();
		RelatednessCalculator calculator = new WuPalmer(bd);
		for(String class1: enumerateY()) {
			int ŷ = this.dict.get(class1);
			
			for(String class2: enumerateY()) {
				int y = this.dict.get(class2);
				double similarity = calculator.calcRelatednessOfWords(class1, class2);
				if (class1.equals(class2)) {
					this.distances[ŷ][y] = 0;
				} else {
					
				}
			}
		}
	}
	
	@Override
	public double delta(String y1, String y2) {
		if (y1.equals(y2)) {
			return 0;
		}
		return 1;
	}
}
