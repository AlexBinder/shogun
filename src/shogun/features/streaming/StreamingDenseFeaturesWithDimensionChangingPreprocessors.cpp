#include <shogun/features/streaming/StreamingDenseFeaturesWithDimensionChangingPreprocessors.h>

using namespace shogun;

//template<class T> 
CStreamingDenseFeaturesWithDimensionChangingPreprocessors::CStreamingDenseFeaturesWithDimensionChangingPreprocessors()
{

holds_DimensionChangingPreprocessors=new CDynamicObjectArray(1);
SG_REF(holds_DimensionChangingPreprocessors);

init();

}

//template<class T> 
CStreamingDenseFeaturesWithDimensionChangingPreprocessors::~CStreamingDenseFeaturesWithDimensionChangingPreprocessors()
{
SG_UNREF(holds_DimensionChangingPreprocessors);
}

EFeatureClass CStreamingDenseFeaturesWithDimensionChangingPreprocessors::get_feature_class() const
{
  return C_STREAMING_DENSE_WITH_DIMENSION_CHANGING_PREPROCESSORS;
}


EFeatureType CStreamingDenseFeaturesWithDimensionChangingPreprocessors::get_feature_type() const 
{ 									
	return F_DREAL;							
}



//template<class T>
bool CStreamingDenseFeaturesWithDimensionChangingPreprocessors::get_next_example()
{
	bool ret_value;
	ret_value=(bool)parser.get_next_example(current_vector.vector,
			current_vector.vlen, current_label);

	for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->get_num_elements();++pind)
	{

		CDimensionChangingPreprocessor *preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(pind);
		REQUIRE(current_vector.vlen=preproctmp->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
		current_vector=preproctmp->apply_to_feature_vector(current_vector);
		preproctmp=NULL;
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
	CStreamingDenseFeaturesWithDimensionChangingPreprocessors* sf=(CStreamingDenseFeaturesWithDimensionChangingPreprocessors *)df;

	SGVector<float64_t> other_vector=sf->get_vector();

	CDimensionChangingPreprocessor *preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproctmp->dimensioncheck(other_vector);
	if(dimtype==2)
	{
		return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
	}
	else 	
	{
		preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(0);
		dimtype=preproctmp->dimensioncheck(other_vector);
		if(dimtype==1)
		{
			for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->get_num_elements();++pind)
			{

				preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(pind);
				REQUIRE(other_vector.vlen=preproctmp->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
				other_vector=preproctmp->apply_to_feature_vector(other_vector);
				preproctmp=NULL;
			}
			return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
		}
		else
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}
	return -1;

}

float32_t CStreamingDenseFeaturesWithDimensionChangingPreprocessors::dot(SGVector<float64_t> other_vector)
{

	CDimensionChangingPreprocessor *preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproctmp->dimensioncheck(other_vector);
	if(dimtype==2)
	{
		return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
	}
	else 	
	{
		preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(0);
		dimtype=preproctmp->dimensioncheck(other_vector);
		if(dimtype==1)
		{
			for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->get_num_elements();++pind)
			{

				preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(pind);
				REQUIRE(other_vector.vlen=preproctmp->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
				other_vector=preproctmp->apply_to_feature_vector(other_vector);
				preproctmp=NULL;
			}
			return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
		}
		else
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}

	return -1;
}


float32_t CStreamingDenseFeaturesWithDimensionChangingPreprocessors::dense_dot(
		const float32_t* vec2, int32_t vec2_len)
{


	CDimensionChangingPreprocessor *preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproctmp->dimensioncheck2(vec2_len);
	if(dimtype==2)
	{

		float32_t result=0;

		for (int32_t i=0; i<current_vector.vlen; i++)
			result+=current_vector[i]*vec2[i];

		return result;

	}
	else 	
	{
		preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(0);
		dimtype=preproctmp->dimensioncheck2(vec2_len);
		if(dimtype==1)
		{

			float64_t tmpv[vec2_len];
			for(int32_t i=0;i<vec2_len;++i)
			{
				tmpv[i]=vec2[i];
			}

			SGVector<float64_t> other_vector(tmpv,vec2_len);
			for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->get_num_elements();++pind)
			{

				preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(pind);
				REQUIRE(other_vector.vlen=preproctmp->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
				other_vector=preproctmp->apply_to_feature_vector(other_vector);

				preproctmp=NULL;
			}
			return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
		}
		else
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");
		}
	}

	return -1;

}


float64_t CStreamingDenseFeaturesWithDimensionChangingPreprocessors::dense_dot(
		const float64_t* vec2, int32_t vec2_len)
{


	CDimensionChangingPreprocessor *preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproctmp->dimensioncheck2(vec2_len);
	if(dimtype==2)
	{

		float32_t result=0;

		for (int32_t i=0; i<current_vector.vlen; i++)
			result+=current_vector[i]*vec2[i];

		return result;

	}
	else 	
	{
		preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(0);
		dimtype=preproctmp->dimensioncheck2(vec2_len);
		if(dimtype==1)
		{

			float64_t tmpv[vec2_len];
			for(int32_t i=0;i<vec2_len;++i)
			{
				tmpv[i]=vec2[i];
			}
			SGVector<float64_t> other_vector(tmpv,vec2_len);
			for(int32_t pind=0;pind<holds_DimensionChangingPreprocessors->get_num_elements();++pind)
			{

				preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->get_element(pind);
				REQUIRE(other_vector.vlen=preproctmp->get_inputfeaturedimensionality(),"failed: current_vector.vlen=preproc->get_inputfeaturedimensionality()\n");
				other_vector=preproctmp->apply_to_feature_vector(other_vector);

				preproctmp=NULL;
			}
			return SGVector<float64_t>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
		}
		else
		{
			SG_ERROR("dimensionality of other feature does not match ...365\n");

		}
	}
	return -1;
}


void CStreamingDenseFeaturesWithDimensionChangingPreprocessors::add_to_dense_vec(
		float32_t alpha, float32_t* vec2, int32_t vec2_len, bool abs_val)
{
	//ASSERT(vec2_len==current_vector.vlen)


	CDimensionChangingPreprocessor *preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproctmp->dimensioncheck2(vec2_len);
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


	CDimensionChangingPreprocessor *preproctmp=(CDimensionChangingPreprocessor *)holds_DimensionChangingPreprocessors->back();
	int32_t dimtype=preproctmp->dimensioncheck2(vec2_len);
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


CFeatures* CStreamingDenseFeaturesWithDimensionChangingPreprocessors::get_streamed_features(
		index_t num_elements)
{
	SG_DEBUG("entering %s(%p)::get_streamed_features(%d)\n", get_name(), this,
			num_elements);

	/* init matrix empty since num_rows is not yet known */
	SGMatrix<float64_t> matrix;

	for (index_t i=0; i<num_elements; ++i)
	{
		/* check if we run out of data */
		if (!get_next_example())
		{
			SG_WARNING("%s::get_streamed_features(): ran out of streaming "
					"data, reallocating matrix and returning!\n", get_name());

			/* allocating space for data so far */
			SGMatrix<float64_t> so_far(matrix.num_rows, i);

			/* copy */
			memcpy(so_far.matrix, matrix.matrix,
					so_far.num_rows*so_far.num_cols*sizeof(float64_t));

			matrix=so_far;
			break;
		}
		else
		{
			/* allocate matrix memory during first run */
			if (!matrix.matrix)
			{
				SG_DEBUG("%s::get_streamed_features(): allocating %dx%d matrix\n",
						get_name(), current_vector.vlen, num_elements);
				matrix=SGMatrix<float64_t>(current_vector.vlen, num_elements);
			}

			/* get an example from stream and copy to feature matrix */
			SGVector<float64_t> vec=get_vector();

			/* check for inconsistent dimensions */
			if (vec.vlen!=matrix.num_rows)
			{
				SG_ERROR("%s::get_streamed_features(): streamed vectors have "
						"different dimensions. This is not allowed!\n",
						get_name());
			}

			/* copy vector into matrix */
			memcpy(&matrix.matrix[current_vector.vlen*i], vec.vector,
					vec.vlen*sizeof(float64_t));

			/* evtl output vector */
			if (sg_io->get_loglevel()==MSG_DEBUG)
			{
				SG_DEBUG("%d. ", i)
				vec.display_vector("streamed vector");
			}

			/* clean up */
			release_example();
		}

	}

	/* create new feature object from collected data */
	CDenseFeatures<float64_t>* result=new CDenseFeatures<float64_t>(matrix);

	SG_DEBUG("leaving %s(%p)::get_streamed_features(%d) and returning %dx%d "
			"matrix\n", get_name(), this, num_elements, matrix.num_rows,
			matrix.num_cols);

	return result;
}

CFeatures* CStreamingDenseFeaturesWithDimensionChangingPreprocessors::duplicate() const
{

SG_NOTIMPLEMENTED; //
return NULL;

/*	
CStreamingDenseFeaturesWithDimensionChangingPreprocessors *tmp=new CStreamingDenseFeaturesWithDimensionChangingPreprocessors::CStreamingDenseFeatures<float64_t>( this);

	tmp->holds_DimensionChangingPreprocessors=this->holds_DimensionChangingPreprocessors;
	return(tmp);
*/
}

void CStreamingDenseFeaturesWithDimensionChangingPreprocessors::add_DimensionChangingPreprocessor(CDimensionChangingPreprocessor * preprocer)
{
	holds_DimensionChangingPreprocessors->push_back(preprocer);
}


void CStreamingDenseFeaturesWithDimensionChangingPreprocessors::init()
{
	SG_ADD( (CSGObject **) &holds_DimensionChangingPreprocessors, "holds_DimensionChangingPreprocessors", "holds_DimensionChangingPreprocessors",MS_NOT_AVAILABLE);
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
