package upmc.ri.bin;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.imageio.ImageIO;

import upmc.ri.io.ImageNetParser;
import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.RankingInstanciation;
import upmc.ri.struct.model.RankingStructModel;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.struct.training.SGDTrainer;
import upmc.ri.utils.Drawing;

public class Ranking {

	public static void main(String[] args) {
		
		/*TO TEST
		Set<String> categories =  new LinkedHashSet<String>();
		categories.add("taxi");
		categories.add("ambulance");*/
		
		Set<String>categories = ImageNetParser.classesImageNet();
		
		
		
		/* Chargement des données en train et en test */
		DataSet<double[], String> dataset = VisualIndexes.createIndex("./sbow/");
				
		Map<String, DataSet<List<double[]>, RankingOutput>> dataset_memory = new HashMap<>();
		double map = 0;
		/*Training*/
		
		List<STrainingSample<List<double[]>,RankingOutput>> listtrain = new ArrayList();;
		List<STrainingSample<List<double[]>,RankingOutput>> listtest = new ArrayList();;
		
		
		for (String classQuery : categories) {
			System.out.println(classQuery);
			/* Instanciation d'un ranking object et modele */
			DataSet<List<double[]>, RankingOutput> rankedDataset_class = RankingFunctions.convertClassif2Ranking(dataset, classQuery);
			dataset_memory.put(classQuery, rankedDataset_class);
			listtrain.addAll(rankedDataset_class.listtrain);
			listtest.addAll(rankedDataset_class.listtest);
		}
		
		int dimpsi = 250;
		double gamma = 10; 
		double lambda = 10e-6;
		int max_iter = 50;
		
		RankingInstanciation instance = new RankingInstanciation();
		RankingStructModel model = new RankingStructModel(instance, dimpsi);
		model.setInstantiation(instance);
		
		
		/* *OPTION* Création d'un evaluateur */
		Evaluator<List<double[]>, RankingOutput> evaluator = new Evaluator<List<double[]>, RankingOutput>();
		evaluator.setListtrain(listtrain);
		evaluator.setListtest(listtest);
		evaluator.setModel(model);
		
		/* Instanciation d'un objet de trainer */
		SGDTrainer<List<double[]>, RankingOutput> sgdTrainer = new SGDTrainer<List<double[]>, RankingOutput>(evaluator, gamma, lambda, max_iter);
		sgdTrainer.train(listtrain, model);

		
		
		for (String classQuery : categories) {
			System.out.println(classQuery);
			
			DataSet<List<double[]>, RankingOutput> rankedDataset = dataset_memory.get(classQuery);
			
			double rp[][] = RankingFunctions.recalPrecisionCurve(model.predict(rankedDataset.listtest.get(0)));
			int nbPlus = rankedDataset.listtest.get(0).output.getNbPlus();
			BufferedImage im = Drawing.traceRecallPrecisionCurve(nbPlus, rp);
			File f = new File(classQuery + "_RPC.png");
			try {
				ImageIO.write(im, "PNG", f);
			} catch (IOException e) {
				e.printStackTrace();
			}
			double ap = RankingFunctions.averagePrecision(model.predict(rankedDataset.listtest.get(0)));
			map += ap;
			System.out.println("AP score : " + ap);
	}
	map = map/9;
	System.out.println("MAP score : " + map);
}
}
