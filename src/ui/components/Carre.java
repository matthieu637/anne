package ui.components;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.geom.Rectangle2D;


@SuppressWarnings("serial")
public abstract class Carre extends Rectangle2D.Float implements FormeDessinable {

	public Carre(int x, int y, int largeur, int hauteur) {
		super(x * largeur, y * hauteur, largeur, hauteur);
	}

	@Override
	public void paint(Graphics2D g) {
		Color last = g.getColor();
		if (getCouleur() != null) {
			g.setColor(getCouleur());
			Rectangle r = getBounds();
			g.fillRect(r.x + 2, r.y + 2, r.width - 3, r.height - 3);
		}
		g.draw(this);
		if (getCouleur() != null)
			g.setColor(last);
	}

	@Override
	public boolean contient(int x, int y) {
		return contains(x, y);
	}
	
	public Point centre(){
		return new Point((int)getCenterX(), (int)getCenterY());
	}
	
	public Rectangle rectangle(){
		return getBounds();
	}

	public abstract Color getCouleur();
}
