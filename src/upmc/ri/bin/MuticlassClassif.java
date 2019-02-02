package upmc.ri.bin;

import java.util.ArrayList;
import java.util.List;

import upmc.ri.io.ImageNetParser;
import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.MultiClass;
import upmc.ri.struct.instantiation.MultiClassHier;
import upmc.ri.struct.model.LinearStructModel_Ex;
import upmc.ri.struct.training.SGDTrainer;

public class MuticlassClassif {

	public static void main(String[] args) {
		/* Chargement des donnees en train et en test */
		DataSet<double[], String> dataset = VisualIndexes.createIndex("./sbow/");
		System.out.println("Donnees chargees, chef!");
		
		/* Instanciation d'un MultiClass et modele */
		MultiClass instance = new MultiClass();
		int dimpsi = VisualIndexes.PCA_DIM * ImageNetParser.classesImageNet().size();
		System.out.println(dimpsi);
		LinearStructModel_Ex<double[], String> model = new LinearStructModel_Ex<double[], String>(instance, dimpsi);
		model.setInstantiation(instance);
		System.out.println("Instanciation terminee, chef!");
		
		/* *OPTION* Creation d'un evaluateur */
		Evaluator<double[],String> evaluator = new Evaluator<double[], String>();
		evaluator.setListtrain(dataset.listtrain);
		evaluator.setListtest(dataset.listtest);
		evaluator.setModel(model);
		System.out.println("Evaluateur cree, chef!");
		
		/* Instanciation d'un objet de trainer */
		double gamma = 10e-2; 
		double lambda = 10e-6;
		int max_iter = 100;
		SGDTrainer<double[], String> sgdTrainer = new SGDTrainer<double[], String>(evaluator, gamma, lambda, max_iter);
		System.out.println("Entraineur operationel, chef!");
		
		sgdTrainer.train(dataset.listtrain, model);
		System.out.println("Entrainement termine, chef!");
		
		List<String> ychap = new ArrayList<String>(dataset.listtest.size());
		List<String> y = new ArrayList<String>(dataset.listtest.size());
		for(STrainingSample<double[], String> ts: dataset.listtest) {
			ychap.add(model.predict(ts));
			y.add(ts.output);
		}

		/* 3. Train on 0-1 loss and evaluate on hierarchical loss */
		model.setInstantiation(new MultiClassHier());
		evaluator.setModel(model);
		evaluator.evaluate();
		System.out.println("Evaluer performances en utilisant la hierarchical loss: " + String.valueOf(evaluator.getErr_train()));
		System.out.println("Test  Error Hier: " + String.valueOf(evaluator.getErr_test() ));
		
		instance.confusionMatrix(ychap, y);
	}
}
