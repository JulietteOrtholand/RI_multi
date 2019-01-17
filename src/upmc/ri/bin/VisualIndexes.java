package upmc.ri.bin;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import upmc.ri.index.ImageFeatures;
import upmc.ri.index.VIndexFactory;
import upmc.ri.io.ImageNetParser;
import upmc.ri.struct.DataSet;
import upmc.ri.struct.STrainingSample;
import upmc.ri.utils.PCA;

public class VisualIndexes {
	public static int PCA_DIM = 250;
	
    public static DataSet<double[], String> createIndex(String path) {    	
    	Set<String> files = ImageNetParser.classesImageNet();

    	List<STrainingSample<double[],String>> train = new ArrayList<STrainingSample<double[],String>>();
    	List<STrainingSample<double[],String>> test = new ArrayList<STrainingSample<double[],String>>();
    	
    	// Loop through each category
    	for (String name : files) {
    		String filename = path + name + ".txt";
    		int count = 0;
    		
    		try {
				List<ImageFeatures> features = ImageNetParser.getFeatures(filename);
				for (ImageFeatures feature : features) {
					// Extract X,Y
					double[] bow = VIndexFactory.computeBow(feature);
					
					if(count < 800) {
						train.add(new STrainingSample<double[], String>(bow, name));
					} else {
						test.add(new STrainingSample<double[], String>(bow, name));
					}
					count++;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
    	}
    	
    	DataSet<double[], String> dataset = new DataSet<double[], String>(train, test);
    	DataSet<double[], String> datasetPca = PCA.computePCA(dataset, VisualIndexes.PCA_DIM);
    	
    	return datasetPca;
    }
    
    public static double squareSum(double...values) {
    	double result = 0;
    	for (double value:values)
    	   result += value * value;
    	return result;
    }
    
    public static void main(String[] args) throws Exception {
    	VisualIndexes.createIndex("./sbow/");
    }
}
