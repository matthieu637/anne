package ui.components;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;


public class Neurone {

	public final static int LARGEUR = 28;
	public final static int HAUTEUR = 28;

	private Color couleur;
	private int couche;
	private int indice;
	private FormeDessinable d;

	public Neurone(int couche, int indice) {
		this.couche = couche;
		this.indice = indice;
	}

	@SuppressWarnings("serial")
	public void setCarre(int x, int y) {
		d = new Carre(x, y, LARGEUR, HAUTEUR) {
			@Override
			public Color getCouleur() {
				return couleur;
			}
		};
	}

	@SuppressWarnings("serial")
	public void setRond(int x, int y) {
		d = new Rond(x, y, LARGEUR, HAUTEUR) {
			@Override
			public Color getCouleur() {
				return couleur;
			}
		};
	}

	public void paint(Graphics2D g2d) {
		d.paint(g2d);
	}

	public boolean contient(int x, int y) {
		return d.contient(x, y);
	}

	public Color getCouleur() {
		return couleur;
	}

	public void setCouleur(Color couleur) {
		this.couleur = couleur;
	}

	public int getCouche() {
		return couche;
	}

	public int getIndice() {
		return indice;
	}

	public void afficherLiaison(Graphics2D g, Neurone n2) {
		Point p1 = d.centre();
		Point p2 = n2.d.centre();
		g.drawLine(p1.x, p1.y, p2.x, p2.y);
	}
	
	public Rectangle rectangle(){
		return d.rectangle();
	}

	public Point centre(){
		return d.centre();
	}
}
