package ui.tabs;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

import modele.Sigmoid01;
import modele.Simulation;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeriesCollection;

import ui.Config;
import ui.Utils;
import ui.components.Entree;
import ui.components.Neurone;
import ui.components.container.EspaceEntree;
import ui.components.container.EspaceNeurone;

@SuppressWarnings("serial")
public class Reseau extends JPanel {

	private JButton learn;

	public Reseau(final JTabbedPane onglet, int x, int y, List<Integer> format, List<Integer> largeur_couche, final float lrate,
			final int epoch) {
		final Simulation sim = new Simulation(format, new Sigmoid01());

		setLayout(null);
		final EspaceEntree conteneurEntree = new EspaceEntree(x, y, sim.getMlp(), this);
		final NNN conteneurReseau = new NNN(conteneurEntree, largeur_couche, sim);
		conteneurReseau.setLocation(conteneurEntree.getSize().width, 0);

		setPreferredSize(new Dimension(conteneurEntree.getSize().width + conteneurReseau.getSize().width, Math.max(
				conteneurEntree.getSize().height, conteneurReseau.getSize().height) + 40));

		learn = new JButton("Learn");
		learn.setBounds(25, conteneurEntree.getHeight(), 75, 30);
		learn.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				conteneurEntree.ecrireEntree(sim);
				sim.lancer(lrate, epoch, epoch / 75);
				learn.setEnabled(false);

				onglet.setEnabledAt(2, true);

				XYSeriesCollection sc = new XYSeriesCollection(Utils.toXYSeries(sim.getClassifApp(), "corpus apprentissage"));

				JFreeChart chart = ChartFactory.createXYLineChart("Erreur de classification", "époque", "erreur", sc,
						PlotOrientation.VERTICAL, true, true, false);

				onglet.setComponentAt(2, new ChartPanel(chart));

			}
		});
		JButton reset = new JButton("Reset");
		reset.setBounds(110, conteneurEntree.getHeight(), 75, 30);
		reset.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				sim.getMlp().initilise_poids(-1, 1);
				sim.viderDonneApp();
				conteneurEntree.resetEntrees();
				conteneurEntree.repaint();
				conteneurReseau.reset();
				onglet.setEnabledAt(2, false);
				learn.setEnabled(true);
			}
		});

		JPanel legend = new JPanel() {
			@Override
			protected void paintComponent(Graphics g) {
				super.paintComponent(g);

				Graphics2D g2d = (Graphics2D) g;
				if (Config.getInstance().isAntialiasing()) {
					g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
					g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
				}

				g.setColor(Color.YELLOW);
				g.fillRect(0, 2, 35, 26);
				g.setColor(Color.WHITE);
				g.fillRect(42, 2, 35, 26);

				g2d.setComposite(java.awt.AlphaComposite.getInstance(java.awt.AlphaComposite.SRC_OVER, 0.5f));

				g.setColor(Color.BLACK);
				g.drawString("> 0.5", 1, 20);
				g.drawString("< 0.5", 43, 20);
			}
		};
		legend.setBounds(200, conteneurEntree.getHeight(), 100, 30);

		add(conteneurEntree);
		add(conteneurReseau);
		add(learn);
		add(reset);
		add(legend);
	}

	public boolean learnEnable() {
		return learn.isEnabled();
	}
}

@SuppressWarnings("serial")
class NNN extends EspaceNeurone {
	private EspaceEntree conteneurEntree;

	public NNN(EspaceEntree conteneurEntree, List<Integer> largeur_couche, Simulation s) {
		super(s, largeur_couche);
		this.conteneurEntree = conteneurEntree;
	}

	@Override
	protected void afficherPoids(Neurone n) {
		super.afficherPoids(n);
		conteneurEntree.pourNeurone(n);
	}

	public void reset() {
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

		Neurone n = neurones.get(neurones.size() - 2);
		g.drawOval(n.rectangle().x + Entree.LARGEUR / 4, n.rectangle().y + Entree.HAUTEUR / 4, Entree.LARGEUR / 2, Entree.HAUTEUR / 2);
		n = neurones.get(neurones.size() - 1);
		g.drawLine(n.rectangle().x + Entree.LARGEUR / 4, n.rectangle().y + Entree.HAUTEUR / 4, n.rectangle().x + 3 * Entree.LARGEUR / 4,
				n.rectangle().y + 3 * Entree.HAUTEUR / 4);
		g.drawLine(n.rectangle().x + Entree.LARGEUR / 4, n.rectangle().y + 3 * Entree.HAUTEUR / 4,
				n.rectangle().x + 3 * Entree.LARGEUR / 4, n.rectangle().y + Entree.HAUTEUR / 4);

		g.setFont(new Font("Sans Serif", Font.BOLD, 12));
		n = neurones.get(0);

		Color last = g.getColor();
		if (n.getCouleur() != null)
			g.setColor(n.getCouleur().getBlue() < 20 ? Color.WHITE : Color.BLACK);

		g.drawString("X", n.centre().x, n.centre().y - 2);

		g.setColor(last);
		n = neurones.get(1);
		if (n.getCouleur() != null)
			g.setColor(n.getCouleur().getBlue() < 20 ? Color.WHITE : Color.BLACK);
		g.drawString("Y", n.centre().x, n.centre().y - 2);
		g.setColor(last);
	}
}