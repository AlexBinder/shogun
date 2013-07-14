#ifndef _StreamingDenseFeaturesWithDimensionChangingPreprocessors__H__
#define _StreamingDenseFeaturesWithDimensionChangingPreprocessors__H__

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/preprocessor/DimensionChangingPreprocessor.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>

namespace shogun{

//template<class T> 
class CStreamingDenseFeaturesWithDimensionChangingPreprocessors : public CStreamingDenseFeatures<float64_t>
{
public:
	CStreamingDenseFeaturesWithDimensionChangingPreprocessors();
	virtual ~CStreamingDenseFeaturesWithDimensionChangingPreprocessors();



	/**
	 * Return the feature type, depending on T.
	 *
	 * @return Feature type as EFeatureType
	 */
	virtual EFeatureType get_feature_type() const;

	/**
	 * Return the feature class
	 *
	 * @return C_STREAMING_DENSE_WITH_DIMENSION_CHANGING_PREPROCESSORS
	 */
	virtual EFeatureClass get_feature_class() const;



	/**
	 * Return the name.
	 *
	 * @return StreamingDenseFeaturesWithDimensionChangingPreprocessors
	 */
	virtual const char* get_name() const
	{
		return "StreamingDenseFeaturesWithDimensionChangingPreprocessors";
	}

	/**
	 * Instructs the parser to return the next example.
	 *
	 * This example is stored as the current_example in this object.
	 *
	 * @return True on success, false if there are no more
	 * examples, or an error occurred.
	 */
	virtual bool get_next_example();

	/**
	 * Return the current feature vector as an SGVector<T>.
	 *
	 * @return The vector as SGVector<T>
	 */
	SGVector<float64_t> get_vector();

	/**
	 * Dot product using the current vector and another vector, passed as arg.
	 *
	 * @param vec The vector with which to calculate the dot product.
	 *
	 * @return Dot product as a float32_t
	 */
	virtual float32_t dot(SGVector<float64_t> vec);

	/**
	 * Dot product taken with another StreamingDotFeatures object.
	 *
	 * Currently only works if it is a CStreamingDenseFeatures object.
	 * It takes the dot product of the current_vectors of both objects.
	 *
	 * @param df CStreamingDotFeatures object.
	 *
	 * @return Dot product.
	 */
	virtual float32_t dot(CStreamingDotFeatures *df);

	/**
	 * Dot product with another dense vector.
	 *
	 * @param vec2 The dense vector with which to take the dot product.
	 * @param vec2_len length of vector
	 * @return Dot product as a float32_t.
	 */
	virtual float32_t dense_dot(const float32_t* vec2, int32_t vec2_len);

	/**
	 * Dot product with another float64_t type dense vector.
	 *
	 * @param vec2 The dense vector with which to take the dot product.
	 * @param vec2_len length of vector
	 * @return Dot product as a float64_t.
	 */
	virtual float64_t dense_dot(const float64_t* vec2, int32_t vec2_len);

	/**
	 * Add alpha*current_vector to another dense vector.
	 * Takes the absolute value of current_vector if specified.
	 *
	 * @param alpha alpha
	 * @param vec2 vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float32_t alpha, float32_t* vec2,
			int32_t vec2_len, bool abs_val=false);

	/**
	 * Add alpha*current_vector to another float64_t type dense vector.
	 * Takes the absolute value of current_vector if specified.
	 *
	 * @param alpha alpha
	 * @param vec2 vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float64_t alpha, float64_t* vec2,
			int32_t vec2_len, bool abs_val=false);

	/**
	 * Return the size of one T object.
	 *
	 * @return Size of T.
	 */
	virtual int32_t get_size() const;

	/** Returns a CDenseFeatures instance which contains num_elements elements
	 * from the underlying stream
	 *
	 * @param num_elements num elements to save from stream
	 * @return CFeatures object of underlying type, NULL if not enough data
	 */
	virtual CFeatures* get_streamed_features(index_t num_elements);

	/**
	 * Duplicate the object.
	 *
	 * @return a duplicate object as CFeatures*
	 */
	virtual CFeatures* duplicate() const;

//*************************

	void add_DimensionChangingPreprocessor(CDimensionChangingPreprocessor * preprocer);












protected:

CDynamicObjectArray* holds_DimensionChangingPreprocessors;

private:

	void init();

};


}

#endif
