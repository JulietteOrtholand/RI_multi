package upmc.ri.struct.instantiation;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;

public class RankingInstanciation implements IStructInstantiation<List<double[]>, RankingOutput> {
	
	@Override
	public double[] psi(List<double[]> x, RankingOutput y) {
		List<Integer> positioning = y.getPositionningFromRanking();
		List<Integer> doc_pos = new ArrayList<Integer>();
		List<Integer> doc_neg = new ArrayList<Integer>();
		
		List<Integer> labels = y.getLabelsGT();
		for(int i=0; i < labels.size() ; i++) {
			if (labels.get(i) < 0) {
				doc_neg.add(i);
			} else if (labels.get(i) > 0) {
				doc_pos.add(i);
			}
		}
		
		double yij = 0;
		double[] psi = new double[x.get(0).length];
		
		for (int i=0; i < doc_pos.size(); i++) { 
			int posId = doc_pos.get(i);
			
			for (int j=0; j < doc_neg.size(); j++) {
				int negId = doc_neg.get(i);
				
				yij = 0;
				if (positioning.get(posId) > positioning.get(negId))
					yij = -1;
				if (positioning.get(posId) < positioning.get(negId))
					yij = 1;
				
				for (int k= 0; k < psi.length; k++) {
					psi[k] += yij * (x.get(posId)[k] - x.get(negId)[k]);
				}
			}
		}
		return psi;
	}

	@Override
	public double delta(RankingOutput y1, RankingOutput y2) {
		return 1 - RankingFunctions.averagePrecision(y2);
	}

	@Override
	public Set<RankingOutput> enumerateY() {
		return null;
	}
}
