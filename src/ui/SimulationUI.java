package ui;

import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

import modele.Simulation;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeriesCollection;

import ui.components.container.EspaceNeurone;
import ui.tabs.Params;

@SuppressWarnings("serial")
public class SimulationUI extends JFrame {

	public SimulationUI(Simulation s, List<Integer> largeur_couche) {
		assert (largeur_couche.size() == s.getMlp().getCouches().size());
		
		if(s.getMlp().getTailleEntrees() > 150)
			Config.getInstance().setTransparence(0.012F);

		setTitle("Mini Simu");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		JTabbedPane onglet = new JTabbedPane();
		
		JPanel jp = new JPanel();
		Params p = new Params();
		p.addParamsAffichage();
		jp.add(p);
		onglet.addTab("Params", jp);

		JPanel conteneur = new EspaceNeurone(s, largeur_couche);
		onglet.addTab("Poids", conteneur);

		XYSeriesCollection sc = new XYSeriesCollection(Utils.toXYSeries(s.getClassifTest(), "corpus test"));
		sc.addSeries(Utils.toXYSeries(s.getClassifApp(), "corpus apprentissage"));

		JFreeChart chart = ChartFactory.createXYLineChart("Erreur de classification", "époque", "erreur", sc, PlotOrientation.VERTICAL,
				true, true, false);

		onglet.addTab("Courbes", new ChartPanel(chart));
		getContentPane().add(onglet);
		setSize(conteneur.getWidth(), conteneur.getHeight() + 20);
		onglet.setSelectedIndex(1);
	}

	public void afficher() {
		setVisible(true);
	}
}

