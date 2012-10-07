package ui.tabs;

import java.awt.Font;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import ui.Config;

@SuppressWarnings("serial")
public class Params extends JPanel {

	public void addParamsDiscretisation(final JFrame frame, final JTabbedPane onglet) {
		setLayout(new GridLayout(10, 2));

		JLabel lparam = new JLabel("Paramètres du réseau", SwingConstants.CENTER);
		Font italic = lparam.getFont().deriveFont(Font.ITALIC);

		lparam.setFont(italic);

		JLabel llrate = new JLabel("Taux d'apprentissage", SwingConstants.CENTER);
		final JTextField lrate = new JTextField("0.15");
		JLabel lpas = new JLabel("Nombre d'époques", SwingConstants.CENTER);
		final JTextField pas = new JTextField("5000");
		JLabel ldiscret = new JLabel("Discrétisation", SwingConstants.CENTER);
		final JTextField discretX = new JTextField("10");
		JLabel ldiscret2 = new JLabel(" x ", SwingConstants.CENTER);
		final JTextField discretY = new JTextField("10");
		JLabel lforme = new JLabel("Neurone par couche", SwingConstants.CENTER);
		final JTextField forme = new JTextField("3,4");
		JLabel laff = new JLabel("Affichage du réseau", SwingConstants.CENTER);
		final JTextField aff = new JTextField("1,1");
		JButton valider = new JButton("Valider");
		valider.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				float learnrate = Float.parseFloat(lrate.getText());
				int epoch = Integer.parseInt(pas.getText());
				int x = Integer.parseInt(discretX.getText());
				int y = Integer.parseInt(discretY.getText());
				List<Integer> format = new LinkedList<Integer>();
				List<Integer> formatAff = new LinkedList<Integer>();

				format.add(2);
				formatAff.add(1);

				StringTokenizer st1 = new StringTokenizer(forme.getText(), ",");
				StringTokenizer st2 = new StringTokenizer(aff.getText(), ",");

				while (st1.hasMoreTokens())
					format.add(Integer.parseInt(st1.nextToken()));

				while (st2.hasMoreTokens())
					formatAff.add(Integer.parseInt(st2.nextToken()));

				format.add(2);
				formatAff.add(1);

				final Reseau r = new Reseau(onglet, x, y, format, formatAff, learnrate, epoch);
				onglet.setComponentAt(1, r);
				onglet.setEnabledAt(1, true);

				// petit hack pour charger les graphismes du prochain onglet dès la pression du bouton valider
				SwingUtilities.invokeLater(new Runnable() {
					@Override
					public void run() {
						BufferedImage image = new BufferedImage(r.getWidth(), r.getHeight(), BufferedImage.TYPE_INT_ARGB);
						Graphics g = image.getGraphics();
						r.paint(g);
						g.dispose();
					}
				});

			}
		});

		add(lparam);
		add(new JPanel());

		add(llrate);
		add(lrate);
		add(lpas);
		add(pas);

		JPanel line3 = new JPanel(new GridLayout());
		add(ldiscret);
		line3.add(discretX);
		line3.add(ldiscret2);
		line3.add(discretY);
		add(line3);

		add(lforme);
		add(forme);

		add(laff);
		add(aff);

		add(new JPanel());
		add(valider);

		addParamsAffichage();
	}

	public void addParamsAffichage() {
		if (!(this.getLayout() instanceof GridLayout))
			setLayout(new GridLayout(3, 2));

		JLabel lparamaff = new JLabel("Paramètres d'affichage", SwingConstants.CENTER);
		Font italic = lparamaff.getFont().deriveFont(Font.ITALIC);
		lparamaff.setFont(italic);

		add(lparamaff);
		add(new JPanel());

		JLabel ltransparence = new JLabel("Transparence des liaisons", SwingConstants.CENTER);
		add(ltransparence);
		final JSlider transparence = new JSlider(0, 100);
		transparence.setValue((int) (Config.getInstance().getTransparence() * 100));
		transparence.addChangeListener(new ChangeListener() {

			@Override
			public void stateChanged(ChangeEvent e) {
				Config.getInstance().setTransparence((float) transparence.getValue() / 100);
				repaint();
			}
		});
		add(transparence);

		JLabel laa = new JLabel("Antialiasing", SwingConstants.CENTER);
		add(laa);
		final JCheckBox aa = new JCheckBox();
		aa.setSelected(Config.getInstance().isAntialiasing());
		aa.addChangeListener(new ChangeListener() {

			@Override
			public void stateChanged(ChangeEvent e) {
				Config.getInstance().setAntialiasing(aa.isSelected());
				repaint();
			}
		});
		add(aa);
	}
}
