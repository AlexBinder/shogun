#include <shogun/features/streaming/StreamingDenseFeaturesWithDimensionChangingPreprocessors.h>

using namespace shogun;

//template<class T> 
CStreamingDenseFeaturesWithDimensionChangingPreprocessors::CStreamingDenseFeaturesWithDimensionChangingPreprocessors()
{

holds_DimensionChangingPreprocessors=new CDynamicObjectArray(1);
SG_REF(holds_DimensionChangingPreprocessors);

}

template<class T> CStreamingDenseFeaturesWithDimensionChangingPreprocessors::~CStreamingDenseFeaturesWithDimensionChangingPreprocessors()
{
SG_UNREF(holds_DimensionChangingPreprocessors);
}

EFeatureClass template<class T> CStreamingDenseFeaturesWithDimensionChangingPreprocessors::get_feature_class()
{
  return C_STREAMING_DENSE_WITH_DIMENSION_CHANGING_PREPROCESSORS;
}

#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> CStreamingDenseFeaturesWithDimensionChangingPreprocessors<sg_type>::get_feature_type() const \
{									\
	return f_type;							\
}

GET_FEATURE_TYPE(F_BOOL, bool)
GET_FEATURE_TYPE(F_CHAR, char)
GET_FEATURE_TYPE(F_BYTE, uint8_t)
GET_FEATURE_TYPE(F_BYTE, int8_t)
GET_FEATURE_TYPE(F_SHORT, int16_t)
GET_FEATURE_TYPE(F_WORD, uint16_t)
GET_FEATURE_TYPE(F_INT, int32_t)
GET_FEATURE_TYPE(F_UINT, uint32_t)
GET_FEATURE_TYPE(F_LONG, int64_t)
GET_FEATURE_TYPE(F_ULONG, uint64_t)
GET_FEATURE_TYPE(F_SHORTREAL, float32_t)
GET_FEATURE_TYPE(F_DREAL, float64_t)
GET_FEATURE_TYPE(F_LONGREAL, floatmax_t)
#undef GET_FEATURE_TYPE


//template<class T>
bool CStreamingDenseFeaturesWithDimensionChangingPreprocessors::get_next_example()
{
	bool ret_value;
	ret_value=(bool)parser.get_next_example(current_vector.vector,
			current_vector.vlen, current_label);

	for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->size();++pind)
	{

		CDimensionChangingPreprocessor *preproc=holds_DimensionChangingPreprocessors->get_element(pind);
		REQUIRE(current_vector.vlen=preproc->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
		current_vector=preproc->apply_to_feature_vector(current_vector);
		preproc=NULL;
	}

	return ret_value;
}

SGVector<float64_t> CStreamingDenseFeaturesWithDimensionChangingPreprocessors::get_vector()
{
	return current_vector;
}


float32_t CStreamingDenseFeaturesWithDimensionChangingPreprocessors::dot(CStreamingDotFeatures* df)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CStreamingDenseFeatures<T>* sf=(CStreamingDenseFeatures<T>*)df;

	SGVector<float64_t> other_vector=sf->get_vector();

	CDimensionChangingPreprocessor *preproc=holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproc->dimensioncheck(other_vector);
	if(dimtype==2)
	{
		return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
	}
	else 	
	{
		preproc=holds_DimensionChangingPreprocessors->get_element(0);
		dimtype=preproc->dimensioncheck(other_vector);
		if(dimtype==1)
		{
			for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->size();++pind)
			{

				preproc=holds_DimensionChangingPreprocessors->get_element(pind);
				REQUIRE(other_vector.vlen=preproc->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
				other_vector=preproc->apply_to_feature_vector(other_vector);
				preproc=NULL;
			}
			return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
		}
		else
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}

}

float32_t CStreamingDenseFeaturesWithDimensionChangingPreprocessors::dot(SGVector<float64_t> other_vector)
{

	CDimensionChangingPreprocessor *preproc=holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproc->dimensioncheck(other_vector);
	if(dimtype==2)
	{
		return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
	}
	else 	
	{
		preproc=holds_DimensionChangingPreprocessors->get_element(0);
		dimtype=preproc->dimensioncheck(other_vector);
		if(dimtype==1)
		{
			for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->size();++pind)
			{

				preproc=holds_DimensionChangingPreprocessors->get_element(pind);
				REQUIRE(other_vector.vlen=preproc->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
				other_vector=preproc->apply_to_feature_vector(other_vector);
				preproc=NULL;
			}
			return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
		}
		else
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}

}


float32_t CStreamingDenseFeaturesWithDimensionChangingPreprocessors::dense_dot(
		const float32_t* vec2, int32_t vec2_len)
{


	CDimensionChangingPreprocessor *preproc=holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproc->dimensioncheck2(vec2_len);
	if(dimtype==2)
	{

		float32_t result=0;

		for (int32_t i=0; i<current_vector.vlen; i++)
			result+=current_vector[i]*vec2[i];

		return result;

	}
	else 	
	{
		preproc=holds_DimensionChangingPreprocessors->get_element(0);
		dimtype=preproc->dimensioncheck2(vec2_len);
		if(dimtype==1)
		{

			float64_t tmpv[vec2_len];
			for(int32_t i=0;i<vec2_len;++i)
			{
				tmpv[i]=vec2[i];
			}

			SGVector<float64_t> other_vector(tmpv,vec2_len);
			for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->size();++pind)
			{

				preproc=holds_DimensionChangingPreprocessors->get_element(pind);
				REQUIRE(other_vector.vlen=preproc->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
				other_vector=preproc->apply_to_feature_vector(other_vector);

				preproc=NULL;
			}
			return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
		}
		else
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}



}


float32_t CStreamingDenseFeaturesWithDimensionChangingPreprocessors::dense_dot(
		const float64_t* vec2, int32_t vec2_len)
{


	CDimensionChangingPreprocessor *preproc=holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproc->dimensioncheck2(vec2_len);
	if(dimtype==2)
	{

		float32_t result=0;

		for (int32_t i=0; i<current_vector.vlen; i++)
			result+=current_vector[i]*vec2[i];

		return result;

	}
	else 	
	{
		preproc=holds_DimensionChangingPreprocessors->get_element(0);
		dimtype=preproc->dimensioncheck2(vec2_len);
		if(dimtype==1)
		{

			SGVector<float64_t> other_vector(vec2,vec2_len);
			for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->size();++pind)
			{

				preproc=holds_DimensionChangingPreprocessors->get_element(pind);
				REQUIRE(other_vector.vlen=preproc->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
				other_vector=preproc->apply_to_feature_vector(other_vector);

				preproc=NULL;
			}
			return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
		}
		else
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}

}


void CStreamingDenseFeaturesWithDimensionChangingPreprocessors::add_to_dense_vec(
		float32_t alpha, float32_t* vec2, int32_t vec2_len, bool abs_val)
{
	//ASSERT(vec2_len==current_vector.vlen)


	CDimensionChangingPreprocessor *preproc=holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproc->dimensioncheck2(vec2_len);
	if(dimtype==2)
	{

		if (abs_val)
		{
			for (int32_t i=0; i<current_vector.vlen; i++)
				vec2[i]+=alpha*CMath::abs(current_vector[i]);
		}
		else
		{
			for (int32_t i=0; i<current_vector.vlen; i++)
				vec2[i]+=alpha*current_vector[i];
		}
	}
	else 	
	{
		//processing of other vector makes no sense here because of its fixed length
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}

}


void CStreamingDenseFeaturesWithDimensionChangingPreprocessors::add_to_dense_vec(
		float64_t alpha, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	//ASSERT(vec2_len==current_vector.vlen)


	CDimensionChangingPreprocessor *preproc=holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproc->dimensioncheck2(vec2_len);
	if(dimtype==2)
	{

		if (abs_val)
		{
			for (int32_t i=0; i<current_vector.vlen; i++)
				vec2[i]+=alpha*CMath::abs(current_vector[i]);
		}
		else
		{
			for (int32_t i=0; i<current_vector.vlen; i++)
				vec2[i]+=alpha*current_vector[i];
		}
	}
	else 	
	{
		//processing of other vector makes no sense here because of its fixed length
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}

}

int32_t CStreamingDenseFeaturesWithDimensionChangingPreprocessors::get_size() const
{
	return sizeof(float64_t);
}


//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<bool> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<char> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<int8_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<uint8_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<int16_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<uint16_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<int32_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<uint32_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<int64_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<uint64_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<float32_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<float64_t> ;
//template class CStreamingDenseFeaturesWithDimensionChangingPreprocessors<floatmax_t> ;
