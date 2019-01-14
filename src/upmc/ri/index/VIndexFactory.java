package upmc.ri.index;

import upmc.ri.utils.VectorOperations;

public class VIndexFactory {

    public static double[] computeBow(ImageFeatures ib) {
    	
    	double[] counter = new double[ImageFeatures.tdico];
    	
		for (Integer w : ib.words)
		    counter[w] += 1;
		
		// Normalisation l2
		double norm = VectorOperations.norm(counter);
		for (int i=0; i < counter.length; i++)
			counter[i] = counter[i] / norm;
		
    	return counter; 
    }
}