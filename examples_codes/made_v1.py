from scipy.stats import norm
from numpy import linspace

def made_v1(x1):
	loc1 = -2
	scale1 = 1.0

	loc2 = 3.5
	scale2 = 0.2

	def pdf_v1(x):
		return norm.pdf(x, loc=loc1, scale=scale1)

	def pdf_v2(x):
		return norm.pdf(x, loc=loc2, scale=scale2)

	return pdf_v2(x1) + pdf_v1(x1)


def main(job_id, params):
	return made_v1(params['x1'])