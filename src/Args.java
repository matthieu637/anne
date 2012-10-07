import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import modele.FonctionTransfert;
import modele.Sigmoid;
import modele.Sigmoid01;
import modele.Simulation;
import modele.TanH;
import ui.DiscretisationEspaceEntree;
import ui.SimulationUI;

public class Args {

	private static final String USAGE = "utilisation : \n"
			+ "\t-x pour lancer une discrétisation de l'espace d'entree\n"
			+ "\t<liste taille couche> <liste taille affichage couche> <fonction transfert> <fichier corpus> <taux apprentissage> <epoques> <precision courbe>\n"
			+ "exemple : \n" + "\tjava -jar mlp-simu.jar -x\n"
			+ "\tjava -jar mlp-simu.jar 256,50,10 16,5,1 sigmoid01 digits.corpus 0.15 55000 250 \n";

	public static void main(String[] args) throws IOException {
		if (args.length == 0)
			System.out.println(USAGE);
		else {
			if (args[0].equals("-x"))
				new DiscretisationEspaceEntree();
			else {
				if (args.length != 7)
					System.out.println(USAGE);
				else {
					String[] tmp = args[0].split(",");
					List<Integer> taille_couche = new ArrayList<Integer>(tmp.length);
					for (String s : tmp)
						taille_couche.add(Integer.parseInt(s));
					tmp = args[1].split(",");
					List<Integer> taille_couche_affichage = new ArrayList<Integer>(tmp.length);
					for (String s : tmp)
						taille_couche_affichage.add(Integer.parseInt(s));

					if (taille_couche_affichage.size() != taille_couche.size()) {
						System.out.println("<liste taille couche> <liste taille affichage couche> doivent avoir le meme nombre d'element");
						System.out.println(USAGE);
					} else {
						if (!args[2].equalsIgnoreCase("sigmoid") && !args[2].equalsIgnoreCase("sigmoid01")
								&& !args[2].equalsIgnoreCase("tanh")) {
							System.out.println("<fonction transfert> ne peut que prendre les valeurs : sigmoid, sigmoid01, tanh");
							System.out.println(USAGE);
						} else {
							FonctionTransfert f = null;
							if (args[2].equalsIgnoreCase("sigmoid"))
								f = new Sigmoid();
							else if (args[2].equalsIgnoreCase("tanh"))
								f = new TanH();
							else
								f = new Sigmoid01();

							Simulation s = new Simulation(taille_couche, f);
							s.corpusDepuis(args[3]);
							s.lancer(Float.parseFloat(args[4]), Integer.parseInt(args[5]), Integer.parseInt(args[6]));
							new SimulationUI(s, taille_couche_affichage).afficher();
						}
					}
				}
			}
		}
	}
}
