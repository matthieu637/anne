package ui.components.container;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JPanel;

import modele.MLP;
import modele.Simulation;
import ui.Config;
import ui.components.Neurone;

@SuppressWarnings("serial")
public class EspaceNeurone extends JPanel {
	protected List<Neurone> neurones;
	private MLP mlp;
	private Neurone selectionne;

	public EspaceNeurone(Simulation s, List<Integer> largeur_couche) {
		this.mlp = s.getMlp();
		neurones = new LinkedList<>();
		setLayout(null);

		// largeur d'une couche
		int largeur = largeur_couche.get(0);
		// indices de départ
		int yBase = 1;
		int xBase = 1;

		int y = yBase;
		int yMax = y;
		// couche d'entrée
		while ((y - yBase) * largeur < mlp.getTailleEntrees()) {
			for (int x = xBase; x < largeur + xBase; x++) {
				Neurone n = new Neurone(0, (y - yBase) * largeur + (x - xBase));
				n.setCarre(x, y);
				neurones.add(n);
			}
			y++;
		}

		yMax = y;
		xBase += 2 + largeur;

		// les autres couches
		for (int couche = 1; couche < mlp.getCouches().size(); couche++) {
			largeur = largeur_couche.get(couche);
			y = yBase;
			while ((y - yBase) * largeur < mlp.getCouches().get(couche).size()) {
				for (int x = xBase; x < largeur + xBase; x++) {
					Neurone n = new Neurone(couche, (y - yBase) * largeur + (x - xBase));
					n.setRond(x, y);
					neurones.add(n);
				}
				y++;
			}
			if (y > yMax)
				yMax = y;
			xBase += 2 + largeur;
		}

		setBounds(0, 0, xBase * Neurone.LARGEUR, (yMax + 2) * Neurone.HAUTEUR);
		MouseAdapter ma = new MouseAdapter() {
			public void mouseDragged(MouseEvent e) {
				// may be improved
				for (Neurone n : neurones)
					if (n.contient(e.getX(), e.getY())) {
						if (n.getCouche() == 0)
							break;
						if (n == selectionne)
							break;

						resetColors();
						n.setCouleur(Color.RED);
						setSelectionne(n);
						afficherPoids(n);
						repaint();
						break;
					}
			};

			public void mouseReleased(MouseEvent e) {
				mouseDragged(e);
			}
		};

		addMouseMotionListener(ma);
		addMouseListener(ma);
	}

	public void setSelectionne(Neurone selectionne) {
		this.selectionne = selectionne;
	}

	protected void afficherPoids(Neurone n) {
		double[] poids = mlp.poidsEntree(n.getCouche(), n.getIndice());

		normalise(poids, 0, mlp.getTailleEntrees());

		int indice = mlp.getTailleEntrees();
		for (int i = 1; i < mlp.getCouches().size() - 1; i++) {
			normalise(poids, indice, indice + mlp.getCouches().get(i).size());
			indice += mlp.getCouches().get(i).size();
		}
	}

	protected void normalise(double[] poids, int debut, int fin) {
		// normalise 0-255
		if (debut >= poids.length)
			return;

		double moy = 0L;
		double variance = 0L;
		for (int i = debut; i < fin; i++) {
			moy += poids[i];
			variance += poids[i] * poids[i];
		}
		moy /= (fin - debut);
		variance /= (fin - debut);
		variance -= moy * moy;

		for (int i = debut; i < fin; i++) {
			int gris = 255 - (int) (((Math.atan((poids[i] - moy) / Math.sqrt(variance)) + Math.PI / 2) / Math.PI) * 255);
			neurones.get(i).setCouleur(new Color(gris, gris, gris));
		}
	}

	protected void resetColors() {
		for (Neurone n : neurones)
			n.setCouleur(null);
	}

	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);

		Graphics2D g2d = (Graphics2D) g;
		if (Config.getInstance().isAntialiasing()) {
			g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
		}

		for (Neurone n : neurones)
			n.paint(g2d);

		g2d.setComposite(java.awt.AlphaComposite.getInstance(java.awt.AlphaComposite.SRC_OVER, Config.getInstance().getTransparence()));

		if (selectionne == null) {
			int index = 0;
			for (int i = 0; i < mlp.getTailleEntrees(); i++)
				for (int j = 0; j < mlp.getCouches().get(1).size(); j++) {
					Neurone n1 = neurones.get(i);
					Neurone n2 = neurones.get(mlp.getTailleEntrees() + j);
					n1.afficherLiaison(g2d, n2);
				}
			index += mlp.getTailleEntrees();
			for (int c = 1; c < mlp.getCouches().size() - 1; c++) {
				for (int i = 0; i < mlp.getCouches().get(c).size(); i++)
					for (int j = 0; j < mlp.getCouches().get(c + 1).size(); j++) {
						Neurone n1 = neurones.get(index + i);
						Neurone n2 = neurones.get(index + mlp.getCouches().get(c).size() + j);
						n1.afficherLiaison(g2d, n2);
					}
				index += mlp.getCouches().get(c).size();
			}
		} else {
			if (selectionne.getCouche() == 1) {
				for (int i = 0; i < mlp.getTailleEntrees(); i++) {
					Neurone n1 = neurones.get(i);
					n1.afficherLiaison(g2d, selectionne);
				}
			} else {
				int index = 0;
				for (int i = 0; i < mlp.getTailleEntrees(); i++)
					for (int j = 0; j < mlp.getCouches().get(1).size(); j++) {
						Neurone n1 = neurones.get(i);
						Neurone n2 = neurones.get(mlp.getTailleEntrees() + j);
						n1.afficherLiaison(g2d, n2);
					}
				index += mlp.getTailleEntrees();
				for (int c = 1; c < selectionne.getCouche() - 1; c++) {
					for (int i = 0; i < mlp.getCouches().get(c).size(); i++)
						for (int j = 0; j < mlp.getCouches().get(c + 1).size(); j++) {
							Neurone n1 = neurones.get(index + i);
							Neurone n2 = neurones.get(index + mlp.getCouches().get(c).size() + j);
							n1.afficherLiaison(g2d, n2);
						}
					index += mlp.getCouches().get(c).size();
				}
				for (int i = 0; i < mlp.getCouches().get(selectionne.getCouche() - 1).size(); i++) {
					Neurone n1 = neurones.get(index + i);
					n1.afficherLiaison(g2d, selectionne);
				}
			}
		}

		g2d.setComposite(java.awt.AlphaComposite.getInstance(java.awt.AlphaComposite.SRC_OVER, 1.0f));
	}
}