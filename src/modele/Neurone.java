package modele;

/**
 * Représente un neurone dans un réseau de perceptron multicouche
 */
public class Neurone {

	/**
	 * Ensemble de poids entrant du neurone
	 */
	private double[] poids;

	/**
	 * Le poids du biais ( neurone supplémentaire toujours à 1 )
	 */
	private double biais;

	/**
	 * Créer un neurone avec nombre_entrees de poids
	 * 
	 * @param nombre_entrees
	 */
	public Neurone(int nombre_entrees) {
		assert (nombre_entrees > 0);

		poids = new double[nombre_entrees];
	}

	/**
	 * @param indice
	 * @return le poids pour l'indice donné
	 */
	public double poid(int indice) {
		assert (indice >= 0 && indice < poids.length);
		return poids[indice];
	}

	/**
	 * @param entrees
	 * @return la somme pondéré des entrées par les poids du neurone (sans lui appliquer de fonction de
	 *         transition)
	 */
	public double sommePondere(Double[] entrees) {
		assert (entrees != null && entrees.length == poids.length);
		double a = 0.;
		for (int i = 0; i < poids.length; i++)
			a += entrees[i] * poids[i];

		a += biais;
		return a;
	}

	/**
	 * Retourne l'état de sortie du neurone pour les entrées données
	 * 
	 * @param entrees
	 * @param f
	 * @return g(a)=g(∑weights[i]×inputs[i])
	 */
	public double calculSortie(Double[] entrees, FonctionTransfert f) {
		return f.calc(sommePondere(entrees));
	}

	/**
	 * Met à jour les poids du neurone
	 * 
	 * wj(t+1)=wj(t)+learning_rate×error×inputsj
	 * 
	 * @param erreur
	 * @param inputs
	 */
	public void apprend(double erreur, Double[] inputs, double taux_apprentissage) {
		for (int i = 0; i < poids.length; i++)
			poids[i] += taux_apprentissage * erreur * inputs[i];

		biais += taux_apprentissage * erreur;
	}

	/**
	 * Initialise aléatoirement les poids du neurone entre min et max
	 * 
	 * @param min
	 * @param max
	 */
	public void initilise_poids(double min, double max) {
		for (int i = 0; i < poids.length; i++)
			poids[i] = Utils.random(min, max);
		biais = Utils.random(min, max);
	}

	/**
	 * @return le poid du biais
	 */
	public double biais() {
		return biais;
	}

	/**
	 * @return ensemble des poids
	 */
	public double[] poids() {
		return poids;
	}
}
