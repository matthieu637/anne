package ui.components.container;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import ui.Config;
import ui.components.Entree;
import ui.components.Neurone;
import ui.tabs.Reseau;

import modele.MLP;
import modele.Sigmoid01;
import modele.Simulation;

@SuppressWarnings("serial")
public class EspaceEntree extends JPanel {
	private List<Entree> grille;
	private MLP mlp;

	public EspaceEntree(int xT, int yT, MLP mlp, final Reseau reseau) {
		this.mlp = mlp;
		grille = new LinkedList<>();
		int x = 1;
		int y = 1;
		for (; y < yT + 1; y++)
			for (x = 1; x < xT + 1; x++)
				grille.add(new Entree(x, y));
		setBounds(0, 0, (x + 2) * Entree.LARGEUR, (y + 1) * Entree.HAUTEUR);

		MouseAdapter ma = new MouseAdapter() {
			@Override
			public void mouseDragged(MouseEvent e) {
				if (reseau.learnEnable())
					for (Entree n : grille)
						if (n.contains(e.getX(), e.getY())) {
							n.setRond(SwingUtilities.isLeftMouseButton(e));
							repaint();
							break;
						}
			}

			public void mouseReleased(MouseEvent e) {
				mouseDragged(e);
			}
		};

		addMouseListener(ma);
		addMouseMotionListener(ma);
	}

	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);

		Graphics2D g2d = (Graphics2D) g;
		if (Config.getInstance().isAntialiasing()) {
			g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
		}

		for (Entree n : grille) {
			n.paint((Graphics2D) g);

			if (n.isRond())
				g.drawOval(n.getBounds().x + Entree.LARGEUR / 4, n.getBounds().y + Entree.HAUTEUR / 4, Entree.LARGEUR / 2,
						Entree.HAUTEUR / 2);
			else {
				g.drawLine(n.getBounds().x + Entree.LARGEUR / 4, n.getBounds().y + Entree.HAUTEUR / 4, n.getBounds().x + 3 * Entree.LARGEUR
						/ 4, n.getBounds().y + 3 * Entree.HAUTEUR / 4);
				g.drawLine(n.getBounds().x + Entree.LARGEUR / 4, n.getBounds().y + 3 * Entree.HAUTEUR / 4, n.getBounds().x + 3
						* Entree.LARGEUR / 4, n.getBounds().y + Entree.HAUTEUR / 4);
			}
		}

		Rectangle b = grille.get(grille.size() - 1).getBounds();
		g.drawString("X", b.x + b.width + 5, b.y + b.height + 15);
		g.drawString("Y", 15, 25);
		g.drawString("0", 15, b.y + b.height + 15);
	}

	public void resetEntrees() {
		for (Entree e : grille) {
			e.setRond(false);
			e.setCouleur(null);
		}
	}

	public void ecrireEntree(Simulation s) {
		for (Entree e : grille) {
			boolean b = e.isRond();
			s.ajouterDonneeApp(Arrays.asList((double) (e.getBounds().x + Neurone.LARGEUR / 2) / Neurone.LARGEUR,
					(double) (e.getBounds().y + Neurone.HAUTEUR / 2) / Neurone.HAUTEUR, b ? 0. : 1., b ? 1. : 0.));
		}
	}

	public void pourNeurone(Neurone n) {
		for (Entree e : grille) {
			Double[] inputs = new Double[2];
			Double[][] outputs;
			inputs[0] = (double) (e.getBounds().x + Neurone.LARGEUR / 2) / Neurone.LARGEUR;
			inputs[1] = (double) (e.getBounds().y + Neurone.HAUTEUR / 2) / Neurone.HAUTEUR;
			outputs = mlp.calculEtatsInterm(inputs, new Sigmoid01());
			if (n.getCouche() < mlp.getCouches().size() - 1)
				e.setCouleur(outputs[n.getCouche()][n.getIndice()] >= 0.5 ? Color.WHITE : Color.YELLOW);
			else {
				e.setCouleur(modele.Utils.index_max(outputs[n.getCouche()]) == n.getIndice() ? Color.WHITE : Color.YELLOW);
			}
		}
		repaint();
	}
}