package modele;

/**
 * Fonction sigmoid sur [0 ; 1]
 */
public class Sigmoid01 implements FonctionTransfert {

	@Override
	public double calc(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	@Override
	public double calcDerivee(double x) {
		return calc(x) * (1 - calc(x));
	}

}
