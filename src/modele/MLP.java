package modele;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Représente un réseau de perceptron multicouche <br/>
 * un neurone d'une couche est connecté à tous les neurones des couches inférieurs et supérieures
 */
public class MLP {

	/**
	 * Liste des couches contenant elles mêmes leurs liste de neurone <br/>
	 * La couche 0 est vide, les entrées n'étant pas représentées par de vrais neurones couches.get(0).size()
	 * n'a donc pas de sens, voir taille_entress
	 */
	private List<List<Neurone>> couches;

	/**
	 * Taille des entrées du réseau. <br/>
	 * Correspond également au nombre de poids d'un neurone de la couche 1
	 */
	private int taille_entrees;

	/**
	 * Crée un perceptron multicouche <br/>
	 * Ne pas oublier d'appeller initilise_poids
	 * 
	 * @param neurones_par_couches
	 *            liste contenant le nombre de neurones désirés par couche, dernier nombre = taille dernière
	 *            couche
	 */
	public MLP(List<Integer> neurones_par_couches) {
		assert (neurones_par_couches != null && neurones_par_couches.size() > 1);

		couches = new ArrayList<List<Neurone>>(neurones_par_couches.size());

		taille_entrees = neurones_par_couches.get(0);
		couches.add(new ArrayList<Neurone>(0));

		for (int i = 1; i < neurones_par_couches.size(); i++) {
			int count = neurones_par_couches.get(i);
			List<Neurone> layer = new ArrayList<Neurone>(count);

			for (int j = 0; j < count; j++)
				layer.add(new Neurone(neurones_par_couches.get(i - 1)));

			couches.add(layer);
		}
	}

	/**
	 * Initialise aléatoirement les poids du réseau entre min et max
	 * 
	 * @param min
	 * @param max
	 */
	public void initilise_poids(double min, double max) {
		assert (min <= max);
		for (int i = 1; i < couches.size(); i++)
			for (Neurone n : couches.get(i))
				n.initilise_poids(min, max);
	}

	/**
	 * Calcule la couche de sortie du réseau
	 * 
	 * @param inputs
	 *            entrées proposées au réseau
	 * @param f
	 *            la fonction d'activation
	 * @return le tableau des valeurs de sortie des neurones de la dernière couche
	 */
	public Double[] calculSortie(Double[] inputs, FonctionTransfert f) {
		return calculEtatsInterm(inputs, f)[couches.size() - 1];
	}

	/**
	 * Apprentissage par descente de gradient
	 * 
	 * @param inputs
	 *            entrée du réseau
	 * @param outputs
	 *            sortie désirée pour l'entrée donnée
	 * @param f
	 *            fonction de transfert
	 * @param learning_rate
	 *            le taux d'apprentissage
	 */
	public void apprendre(Double[] inputs, Double[] outputs, FonctionTransfert f, double learning_rate) {
		assert (inputs != null && outputs != null && inputs.length == taille_entrees && outputs.length == couches.get(couches.size() - 1)
				.size());

		Double[][] interState = calculEtatsInterm(inputs, f);

		double[][] errors = new double[couches.size() - 1][];

		// construire le vector d'erreur de la sortie aux entrées
		int i = couches.size() - 2;
		errors[i] = new double[couches.get(i + 1).size()];

		// -- couche de sortie
		for (int j = 0; j < couches.get(i + 1).size(); j++) {
			Neurone n = couches.get(i + 1).get(j);
			double n_a = n.sommePondere(interState[i]);
			double n_o = f.calc(n_a);
			errors[i][j] = f.calcDerivee(n_a) * (outputs[j] - n_o);
		}

		// -- couches intermédiaires
		for (i--; i >= 0; i--) {
			errors[i] = new double[couches.get(i + 1).size()];
			for (int j = 0; j < couches.get(i + 1).size(); j++) {
				Neurone n = couches.get(i + 1).get(j);
				double n_a = n.sommePondere(interState[i]);
				double wsum = 0.F;
				for (int k = 0; k < couches.get(i + 2).size(); k++)
					wsum += couches.get(i + 2).get(k).poid(j) * errors[i + 1][k];
				errors[i][j] = f.calcDerivee(n_a) * wsum;
			}
		}

		// mise à jour des poids avec le vecteur d'erreur
		for (i = 1; i < couches.size(); i++)
			for (int j = 0; j < couches.get(i).size(); j++)
				couches.get(i).get(j).apprend(errors[i - 1][j], interState[i - 1], learning_rate);

	}

	/**
	 * Détermine les états de sorties de chaque neurone pour chaque couche
	 * 
	 * @param inputs
	 *            entrée du réseau
	 * @param f
	 *            la fonction de transfert
	 * @return la sortie de chaque neurone du réseau pour les entrées données
	 */
	public Double[][] calculEtatsInterm(Double[] inputs, FonctionTransfert f) {
		assert (inputs != null && inputs.length == taille_entrees);

		Double[][] interState = new Double[couches.size()][];

		interState[0] = Arrays.copyOf(inputs, inputs.length);

		int i;
		for (i = 1; i < couches.size(); i++) {
			interState[i] = new Double[couches.get(i).size()];
			for (int j = 0; j < couches.get(i).size(); j++)
				interState[i][j] = couches.get(i).get(j).calculSortie(interState[i - 1], f);
		}

		return interState;
	}

	/**
	 * @return la taille de l'entrée du réseau
	 */
	public int getTailleEntrees() {
		return taille_entrees;
	}

	/**
	 * @return tous les neurones
	 */
	public List<List<Neurone>> getCouches() {
		return couches;
	}

	/**
	 * Retourne un ensemble de valeurs représentant l'importance des neurones ascendants ( en fonction des
	 * poids ) pour un neurone donné
	 * 
	 * @param couchewant
	 *            couche du neurone
	 * @param indicewant
	 *            indice du neurone
	 * @return l'importance des poids de tous les neurones des couches précédentes
	 */
	public double[] poidsEntree(int couchewant, int indicewant) {
		assert (couchewant > 0);

		int taille = taille_entrees;
		for (int i = 1; i < couches.size(); i++)
			if (i < couchewant)
				taille += couches.get(i).size();
			else
				break;

		double[] resultat = new double[taille];
		int couche = 0;
		int indice = 0;
		int i = 0;
		for (; i < taille_entrees; i++)
			resultat[i] = poidsRepr(couche, indice++, couchewant, indicewant);
		couche++;
		indice = 0;
		for (i = taille_entrees; i < resultat.length; i++) {
			resultat[i] = poidsRepr(couche, indice++, couchewant, indicewant);
			if (indice >= couches.get(couche).size()) {
				indice = 0;
				couche++;
			}
		}

		return resultat;
	}

	/**
	 * Fonction récursive qui retourne la somme des poids pondérés entre 2 neurones <br/>
	 * Le second neurone doit être dans une couche supérieure à celui du premier
	 * 
	 * @param couche
	 *            couche du premier neurone
	 * @param indice
	 *            indice du premier neurone
	 * @param couchewant
	 *            indice du second neurone
	 * @param indexwant
	 *            indice du second neurone
	 * @return le poid pondéré entre 2 neurones
	 */
	private double poidsRepr(int couche, int indice, int couchewant, int indexwant) {
		if (couche == couchewant - 1)
			return couches.get(couchewant).get(indexwant).poid(indice);
		else {
			double res = 0;
			for (int i = 0; i < couches.get(couche + 1).size(); i++)
				res += poidsRepr(couche + 1, i, couchewant, indexwant)
						* (couches.get(couche + 1).get(i).poid(indice) + couches.get(couche + 1).get(i).biais());
			return res;
		}
	}
}
