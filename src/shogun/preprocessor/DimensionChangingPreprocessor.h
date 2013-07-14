#ifndef _CDimensionChangingPreprocessor__H__
#define _CDimensionChangingPreprocessor__H__

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/Features.h>
#include <shogun/preprocessor/Preprocessor.h>

namespace shogun{

class CDimensionChangingPreprocessor : public CPreprocessor
{
//Designed for minimally safe use with Streamingfeatures
// allows for check for equality of parameters of other preprocessor for usage in ::dot (CStreamingDotFeatures *df)
// allows to check whether feature dimensionality matches input or output dimensionality in ::dot (SGVector< T > vec) and dense_dot (const float32_t *vec2, int32_t vec2_len) [obviously no check can be made in this case whether they have been created using the same preprocessor parameters if the dinmensioncheck returns the value 2]
	public:

	CDimensionChangingPreprocessor();
	virtual ~CDimensionChangingPreprocessor();

	// checks for equality of parameters and processing coefficients
	virtual bool check_equality_to_other_preprocessor(const CDimensionChangingPreprocessor & otherpreproc)=0;
	virtual bool is_initialized()=0; //checks whether it is initialized
	

	//virtual apply_to_vector(SGVector<T> sgvec1)=0;


 	template<class T>
	int32_t dimensioncheck(SGVector<T> sgvec1);// returns 0 if vector length does NOT match input or output dimensionality
	//returns 1 if if vector length DOES match input  dimensionality
	//returns 2 if if vector length DOES match output dimensionality
	int32_t dimensioncheck2(const int32_t len1); // same for integer vector length

	virtual int32_t get_inputfeaturedimensionality();
	virtual int32_t get_outputfeaturedimensionality();

	virtual void set_inputfeaturedimensionality(const int32_t dim);
	virtual void set_outputfeaturedimensionality(const int32_t dim);
	


	private:

	int32_t m_inputfeaturedimensionality;
	int32_t m_outputfeaturedimensionality;


};

} // namespace shogun
#endif
