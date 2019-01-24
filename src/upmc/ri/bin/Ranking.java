package upmc.ri.bin;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.instantiation.RankingInstanciation;
import upmc.ri.struct.model.RankingStructModel;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.struct.training.SGDTrainer;
import upmc.ri.utils.Drawing;

public class Ranking {

	public static void main(String[] args) {
		//String classQuery = "ambulance";
		//String classQuery = "wood-frog";
		String classQuery = "harp";
		//String classQuery = "european_fire_salamander";
		
		/* Chargement des données en train et en test */
		DataSet<double[], String> dataset = VisualIndexes.createIndex("./sbow/");
		
		/* Instanciation d'un ranking object et modele */
		DataSet<List<double[]>, RankingOutput> rankedDataset = RankingFunctions.convertClassif2Ranking(dataset, classQuery);
		int dimpsi = rankedDataset.listtest.get(0).input.get(0).length;
		
		RankingInstanciation instance = new RankingInstanciation();
		RankingStructModel model = new RankingStructModel(instance, dimpsi);
		model.setInstantiation(instance);
		
		/* *OPTION* Création d'un evaluateur */
		Evaluator<List<double[]>, RankingOutput> evaluator = new Evaluator<List<double[]>, RankingOutput>();
		evaluator.setListtrain(rankedDataset.listtrain);
		evaluator.setListtest(rankedDataset.listtest);
		evaluator.setModel(model);
		
		/* Instanciation d'un objet de trainer */
		double gamma = 10; 
		double lambda = 10e-6;
		int max_iter = 50;
		SGDTrainer<List<double[]>, RankingOutput> sgdTrainer = new SGDTrainer<List<double[]>, RankingOutput>(evaluator, gamma, lambda, max_iter);
		sgdTrainer.train(rankedDataset.listtrain, model);
		
		double rp[][] = RankingFunctions.recalPrecisionCurve(model.predict(rankedDataset.listtest.get(0)));
		int nbPlus = rankedDataset.listtest.get(0).output.getNbPlus();
		BufferedImage im = Drawing.traceRecallPrecisionCurve(nbPlus, rp);
		File f = new File(classQuery + "_RPC.png");
		try {
			ImageIO.write(im, "PNG", f);
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("AP score: " + RankingFunctions.averagePrecision(model.predict(rankedDataset.listtest.get(0))));
	}

}
