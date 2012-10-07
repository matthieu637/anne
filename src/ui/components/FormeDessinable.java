package ui.components;

import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;

public interface FormeDessinable {
	void paint(Graphics2D g);
	boolean contient(int x, int y);
	Point centre();
	Rectangle rectangle();
}
