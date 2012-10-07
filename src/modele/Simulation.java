package modele;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

/**
 * Permet de gérer une simulation complète d'un perceptron multicouche : donnée de test, d'apprentissage, nombre
 * d'époque, données depuis fichier ...
 */
public class Simulation {

	/**
	 * Le réseau de neurone
	 */
	private MLP mlp;

	/**
	 * Les entrées et sorties du corpus d'apprentissage : le réseau les apprend
	 */
	private List<Double[]> corpusApprentissageInputs, corpusApprentissageOutputs;

	/**
	 * Les entrées et sortie du corpus de test : les tests sont effectués dessus et on ne les donne jamais explicitement
	 * au réseau
	 */
	private List<Double[]> corpusTestInputs, corpusTestOutputs;

	/**
	 * Fonction de transfert utilisée lors de l'apprentissage et des tests
	 */
	private FonctionTransfert f;

	/**
	 * Les erreurs de classification sur le corpus d'apprentissage et de test
	 */
	private double[] classifTest, classifApp;

	public Simulation(List<Integer> format, FonctionTransfert f) {
		mlp = new MLP(format);
		mlp.initilise_poids(-0.5, 0.5);
		this.f = f;

		corpusApprentissageInputs = new LinkedList<Double[]>();
		corpusApprentissageOutputs = new LinkedList<Double[]>();
		corpusTestInputs = new LinkedList<Double[]>();
		corpusTestOutputs = new LinkedList<Double[]>();
	}

	/**
	 * Ajoute un couple d'entrée/sortie dans le corpus d'apprentissage
	 * 
	 * @param entre_sortie
	 *            liste [entrées ... sortie], la taille des entrées doit être égale au nombre de neurone d'entrée du
	 *            réseau, de même pour la taille des sorties avec le nombre de neurone de sortie du réseau
	 */
	public void ajouterDonneeApp(List<Double> entre_sortie) {
		Double[] inputs = new Double[mlp.getTailleEntrees()];
		Double[] outputs = new Double[entre_sortie.size() - mlp.getTailleEntrees()];

		entre_sortie.subList(0, inputs.length).toArray(inputs);
		entre_sortie.subList(inputs.length, inputs.length + outputs.length).toArray(outputs);

		corpusApprentissageInputs.add(inputs);
		corpusApprentissageOutputs.add(outputs);
	}

	/**
	 * Ajoute un couple d'entrée/sortie dans le corpus de test
	 * 
	 * @param entre_sortie
	 *            liste [entrées ... sortie], la taille des entrées doit être égale au nombre de neurone d'entrée du
	 *            réseau, de même pour la taille des sorties avec le nombre de neurone de sortie du réseau
	 */
	public void ajouterDonneeTest(List<Double> entre_sortie) {
		Double[] inputs = new Double[mlp.getTailleEntrees()];
		Double[] outputs = new Double[entre_sortie.size() - mlp.getTailleEntrees()];

		entre_sortie.subList(0, inputs.length).toArray(inputs);
		entre_sortie.subList(inputs.length, inputs.length + outputs.length).toArray(outputs);

		corpusTestInputs.add(inputs);
		corpusTestOutputs.add(outputs);
	}

	/**
	 * Lance une simulation : un apprentissage avec les corpus remplis au préalable et des tests tout les pas_test <br/>
	 * A chaque époque, on choisit aléatoirement une donnée dans le corpus d'apprentissage et on la présente au réseau
	 * 
	 * @param taux_apprentissage
	 *            taux d'apprentissage du réseau
	 * @param epoque
	 *            le nombre d'époque ( = nombre de couple entrées/sorties présentées au réseau )
	 * @param pas_test
	 *            détermine à quelle époque il faut déterminer l'erreur de classification sur le corpus entier
	 */
	public void lancer(float taux_apprentissage, int epoque, int pas_test) {
		classifTest = new double[epoque];
		classifApp = new double[epoque];
		Random r = new Random();
		for (int i = 0; i < epoque; i++) {
			int corpusSize = corpusApprentissageInputs.size();
			int isample = r.nextInt(corpusSize);

			if (i % pas_test == 0) {
				classifTest[i] = errorClassifitionTest();
				classifApp[i] = errorClassifitionApp();
			} else {
				classifTest[i] = -1;
				classifApp[i] = -1;
			}
			mlp.apprendre(corpusApprentissageInputs.get(isample), corpusApprentissageOutputs.get(isample), f, taux_apprentissage);
		}
	}

	/**
	 * Lit le corpus depuis un fichier, 25% des données vont dans le corpus de test, le reste dans celui d'apprentissage <br/>
	 * Le format du fichier doit être le suivant : <br/>
	 * - une ligne = un couple entrée/sortie ( = nombre de neurone de la couche 0 + nombre de neurone de la dernière
	 * couche) <br/>
	 * - les élements de la ligne sont séparés par des espaces <br/>
	 * 
	 * 
	 * @param chemin
	 */
	public void corpusDepuis(String chemin) throws IOException {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(chemin));
		} catch (FileNotFoundException e) {
			br = new BufferedReader(new InputStreamReader(Simulation.class.getClassLoader().getResourceAsStream(chemin)));
		}

		String line = null;
		Random r = new Random();

		while ((line = br.readLine()) != null) {
			List<Double> l = new LinkedList<Double>();
			StringTokenizer st = new StringTokenizer(line, " ");

			while (st.hasMoreTokens())
				l.add(Double.parseDouble(st.nextToken()));
			if (r.nextInt(4) > 0)
				ajouterDonneeApp(l);
			else
				ajouterDonneeTest(l);
		}
		br.close();
	}

	/**
	 * Calcul l'erreur de classification sur l'ensemble du corpus de test
	 * 
	 * @return moyenne de l'erreur de classif
	 */
	public double errorClassifitionTest() {
		double error = 0.;
		int i;
		for (i = 0; i < corpusTestInputs.size(); i++) {
			if (Utils.index_max(mlp.calculSortie(corpusTestInputs.get(i), f)) != Utils.index_max(corpusTestOutputs.get(i)))
				error += 1.;
		}
		return error / i;
	}

	/**
	 * Calcul l'erreur de classification sur l'ensemble du corpus d'apprentissage
	 * 
	 * @return moyenne de l'erreur de classif
	 */
	public double errorClassifitionApp() {
		double error = 0.;
		int i;
		for (i = 0; i < corpusApprentissageInputs.size(); i++) {
			if (Utils.index_max(mlp.calculSortie(corpusApprentissageInputs.get(i), f)) != Utils
					.index_max(corpusApprentissageOutputs.get(i)))
				error += 1.;
		}
		return error / i;
	}

	/**
	 * @return le réseau de neurone
	 */
	public MLP getMlp() {
		return mlp;
	}

	/**
	 * @return les erreurs de classification sur le corpus de test
	 */
	public double[] getClassifTest() {
		return classifTest;
	}

	/**
	 * @return les erreurs de classification sur le corpus d'apprentissage
	 */
	public double[] getClassifApp() {
		return classifApp;
	}

	/**
	 * Vide les corpus de test et d'apprentissage
	 */
	public void viderDonneApp() {
		corpusApprentissageInputs.clear();
		corpusApprentissageOutputs.clear();
	}
}
