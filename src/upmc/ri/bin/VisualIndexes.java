package upmc.ri.bin;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import upmc.ri.index.ImageFeatures;
import upmc.ri.index.VIndexFactory;
import upmc.ri.io.ImageNetParser;
import upmc.ri.struct.DataSet;
import upmc.ri.struct.STrainingSample;
import upmc.ri.utils.PCA;

public class VisualIndexes {
    public static DataSet createIndex() {

    	final VIndexFactory factory = new VIndexFactory();
    	ImageNetParser parser = new ImageNetParser();
    	
    	Set<String> files = ImageNetParser.classesImageNet();

    	List<STrainingSample<double[],String>> train = new ArrayList<STrainingSample<double[],String>>();
    	List<STrainingSample<double[],String>> test = new ArrayList<STrainingSample<double[],String>>();
    	int count = 0;
    	// Loop through each category
    	for (Iterator<String> i=files.iterator(); i.hasNext();) {
    		String name = i.next();
    		String filename = "./sbow/" + name + ".txt";
    		
    		try {
				List<ImageFeatures> features = parser.getFeatures(filename);
				for (ImageFeatures feature : features) {
					// Extract X,Y
					double[] bow = factory.computeBow(feature);
					List<Double> x = feature.getX();
					
					if(count<800) {
						train.add(new STrainingSample(bow,name));
					}
					else {
						test.add(new STrainingSample(bow,name));
					}
				}
				
				
			} catch (Exception e) {
				e.printStackTrace();
			}
    	}
    	
    	DataSet dataset = new DataSet(train, test);
    	DataSet datasetPca = PCA.computePCA(dataset , 250);
    	
    	return datasetPca;
    }
    
    public static DataSet<X,Y> convertClasses(DataSet<X,Y> dataset){
    	
    }
    
    public static double squareSum(double...values) {
    	double result = 0;
    	for (double value:values)
    	   result += value * value;
    	return result;
    }
}
