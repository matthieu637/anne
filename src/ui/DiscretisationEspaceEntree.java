package ui;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import ui.tabs.Params;

@SuppressWarnings("serial")
public class DiscretisationEspaceEntree extends JFrame {

	public DiscretisationEspaceEntree() {

		setTitle("Mini Simu");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		final JFrame f = this;

		final JTabbedPane onglet = new JTabbedPane();
		onglet.addChangeListener(new ChangeListener() {
			
			@Override
			public void stateChanged(ChangeEvent e) {
				f.pack();
			}
		});
		JPanel jp = new JPanel();
		Params p = new Params();
		p.addParamsDiscretisation(this, onglet);
		jp.add(p);
		onglet.addTab("Params", jp);
		onglet.addTab("Réseau", null);
		onglet.addTab("Courbes", null);
		onglet.setEnabledAt(1, false);
		onglet.setEnabledAt(2, false);

		getContentPane().add(onglet);
		pack();
		setVisible(true);
	}
}


