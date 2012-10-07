package modele;

/**
 * Fonction de transfert utilisée pour déterminer l'activation d'un neurone
 */
public interface FonctionTransfert {
	
	/**
	 * Fonction permettant de définir l'activation d'un neurone
	 * @param x somme pondérée des entrées par les poids
	 * @return f(x) 
	 */
	double calc(double x);
	
	/**
	 * La dérivé de la fonction de transfert, utilisé pour l'apprentissage du neurone
	 * @param x somme pondérée des entrées par les poids
	 * @return f'(x)
	 */
	double calcDerivee(double x);
}
