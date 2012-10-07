package modele;

/**
 * Regroupe des méthodes générales
 */
public class Utils {

	/**
	 * Retourne l'indice de l'élement le plus grand dans le tableau
	 * 
	 * @param tableau
	 * @return indice
	 */
	public static int index_max(Double[] d) {
		int imax = 0;
		for (int i = 0; i < d.length; i++)
			if (d[imax] < d[i])
				imax = i;

		return imax;
	}

	/**
	 * @param min
	 * @param max
	 * @return un nombre aléatoire entre min et max
	 */
	public static double random(double min, double max) {
		return min + Math.random() * Math.abs(max - min);
	}
}
