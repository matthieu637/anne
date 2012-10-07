import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import modele.FonctionTransfert;
import modele.MLP;
import modele.Sigmoid;
import modele.Sigmoid01;
import modele.Simulation;
import modele.TanH;
import ui.SimulationUI;

public class Exemples {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		// OR();
		// XOR();
		// XOR01();
		// AND();
		 digits();
		//new Separation();
	}

	public static void digits() throws IOException {
		Simulation s = new Simulation(Arrays.asList(16 * 16, 50, 10), new Sigmoid01());
		s.corpusDepuis("digits.corpus");
		s.lancer(0.15F, 55000, 250);
		new SimulationUI(s, Arrays.asList(16, 5, 1)).afficher();
	}

	public static void AND() {
		Simulation s = new Simulation(Arrays.asList(2, 2, 1), new Sigmoid01());
		s.ajouterDonneeApp(Arrays.asList(0., 0., 0.));
		s.ajouterDonneeApp(Arrays.asList(1., 0., 0.));
		s.ajouterDonneeApp(Arrays.asList(0., 1., 0.));
		s.ajouterDonneeApp(Arrays.asList(1., 1., 1.));
		s.lancer(0.15F, 1000 * 4, 1);
		new SimulationUI(s, Arrays.asList(1, 1, 1)).afficher();
		System.out.println(Arrays.toString(s.getMlp().getCouches().get(1).get(0).poids()));
		System.out.println(Arrays.toString(s.getMlp().getCouches().get(1).get(1).poids()));
		System.out.println(s.getMlp().getCouches().get(1).get(0).biais());
		System.out.println(s.getMlp().getCouches().get(1).get(0).biais());
		System.out.println(Arrays.toString(s.getMlp().getCouches().get(2).get(0).poids()));
	}

	public static void OR() {
		double lrate = 0.15;
		FonctionTransfert f = new TanH();

		MLP mlp = new MLP(Arrays.asList(2, 2, 1));
		mlp.initilise_poids(-1., 1.);

		Random r = new Random();
		for (int n = 0; n < 100 * 4; n++) {
			switch (r.nextInt(4)) {
			case 0:
				mlp.apprendre(new Double[] { -1., -1. }, new Double[] { -1. }, f, lrate);
				break;
			case 1:
				mlp.apprendre(new Double[] { -1., 1. }, new Double[] { 1. }, f, lrate);
				break;
			case 2:
				mlp.apprendre(new Double[] { 1., -1. }, new Double[] { 1. }, f, lrate);
				break;
			case 3:
				mlp.apprendre(new Double[] { 1., 1. }, new Double[] { 1. }, f, lrate);
				break;
			}
		}

		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { -1., -1. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { 1., -1. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { -1., 1. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { 1., 1. }, f)));
		System.out.println();
	}

	public static void XOR() {
		double lrate = 0.15;
		FonctionTransfert f = new Sigmoid();

		MLP mlp = new MLP(Arrays.asList(2, 2, 1));
		mlp.initilise_poids(-1., 1.);

		Random r = new Random();
		for (int n = 0; n < 300 * 4; n++) {
			switch (r.nextInt(4)) {
			case 0:
				mlp.apprendre(new Double[] { -1., -1. }, new Double[] { -1. }, f, lrate);
				break;
			case 1:
				mlp.apprendre(new Double[] { -1., 1. }, new Double[] { 1. }, f, lrate);
				break;
			case 2:
				mlp.apprendre(new Double[] { 1., -1. }, new Double[] { 1. }, f, lrate);
				break;
			case 3:
				mlp.apprendre(new Double[] { 1., 1. }, new Double[] { -1. }, f, lrate);
				break;
			}
		}

		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { -1., -1. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { 1., -1. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { -1., 1. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { 1., 1. }, f)));
		System.out.println();
	}

	public static void XOR01() {
		double lrate = 0.15;
		FonctionTransfert f = new Sigmoid01();

		List<Integer> li = Arrays.asList(2, 2, 1);
		MLP mlp = new MLP(li);
		mlp.initilise_poids(-1., 1.);

		Random r = new Random();
		for (int n = 0; n < 4000. * 4; n++) {
			switch (r.nextInt(4)) {
			case 0:
				mlp.apprendre(new Double[] { 0., 0. }, new Double[] { 0. }, f, lrate);
				break;
			case 1:
				mlp.apprendre(new Double[] { 0., 1. }, new Double[] { 1. }, f, lrate);
				break;
			case 2:
				mlp.apprendre(new Double[] { 1., 0. }, new Double[] { 1. }, f, lrate);
				break;
			case 3:
				mlp.apprendre(new Double[] { 1., 1. }, new Double[] { 0. }, f, lrate);
				break;
			}
		}

		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { 0., 0. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { 1., 0. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { 0., 1. }, f)));
		System.out.println(Arrays.toString(mlp.calculSortie(new Double[] { 1., 1. }, f)));
		System.out.println();
	}
}
