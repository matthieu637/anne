package ui;

/**
 * Singleton contenant des informations sur la configuration graphique de l'application
 */
public class Config {

	private static final Config instance = new Config();

	private float transparence = 0.3f;
	private boolean antialiasing = true;

	public static Config getInstance() {
		return instance;
	}

	public float getTransparence() {
		return transparence;
	}

	public void setTransparence(float transparence) {
		this.transparence = transparence;
	}

	public boolean isAntialiasing() {
		return antialiasing;
	}

	public void setAntialiasing(boolean antialiasing) {
		this.antialiasing = antialiasing;
	}
}
