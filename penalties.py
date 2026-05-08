#Threshold operators

def soft_threshold(z, lam):
    return np.sign(z) * max(abs(z) - lam, 0.0)


def scad_threshold(z, lam, a=3.7):
    az = abs(z)
    s = np.sign(z)

    if az <= lam:
        return 0.0
    elif az <= 2 * lam:
        return s * (az - lam)
    elif az <= a * lam:
        return s * ((a - 1) * az - a * lam) / (a - 2)
    else:
        return z


def mcp_threshold(z, lam, gamma=3.0):
    az = abs(z)
    s = np.sign(z)

    if az <= gamma * lam:
        return s * max(az - lam, 0.0) / (1 - 1 / gamma)
    else:
        return z
