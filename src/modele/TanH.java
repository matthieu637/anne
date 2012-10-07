package modele;

/**
 *	Fonction tangente hyperbolique sur [0 ; 1]
 */
public class TanH implements FonctionTransfert {

	@Override
	public double calc(double x) {
		return Math.tanh(x);
	}

	@Override
	public double calcDerivee(double x) {
		return 1 - Math.tanh(x) * Math.tanh(x);
	}

}
