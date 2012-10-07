package ui;

import org.jfree.data.xy.XYSeries;

public class Utils {
	public static XYSeries toXYSeries(double d[], String title) {
		XYSeries s = new XYSeries(title);
		for (int i = 0; i < d.length; i++)
			if (d[i] != -1)
				s.add(i, d[i]);

		return s;
	}
}
