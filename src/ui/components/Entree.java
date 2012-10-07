package ui.components;

import java.awt.Color;


@SuppressWarnings("serial")
public class Entree extends Carre {

	public final static int LARGEUR = 28;
	public final static int HAUTEUR = 28;

	private boolean rond;
	private Color couleur;

	public Entree(int x, int y) {
		super(x, y, LARGEUR, HAUTEUR);
		rond = false;
	}

	public void setRond(boolean b) {
		rond = b;
	}

	public boolean isRond() {
		return rond;
	}

	public void setCouleur(Color color) {
		couleur = color;
	}

	public Color getCouleur() {
		return couleur;
	}
}
