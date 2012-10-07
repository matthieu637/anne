package modele;

/**
 * Fonction sigmoid sur [-1 ; 1]
 */
public class Sigmoid implements FonctionTransfert {

	@Override
	public double calc(double x) {
		return (Math.exp(x) - 1) / (1 + Math.exp(x));
	}

	@Override
	public double calcDerivee(double x) {
		return (2 * Math.exp(x)) / ((Math.exp(x) + 1) * (Math.exp(x) + 1));
	}

}
