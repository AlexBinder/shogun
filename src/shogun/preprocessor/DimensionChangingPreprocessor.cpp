#include <shogun/preprocessor/DimensionChangingPreprocessor.h>

using namespace shogun;

CDimensionChangingPreprocessor::CDimensionChangingPreprocessor():m_inputfeaturedimensionality(-1),m_outputfeaturedimensionality(-1)
{

}

CDimensionChangingPreprocessor::~CDimensionChangingPreprocessor()
{
	//does nothing yet
}

int32_t CDimensionChangingPreprocessor::get_inputfeaturedimensionality()
{
	return m_inputfeaturedimensionality;
}

int32_t CDimensionChangingPreprocessor::get_outputfeaturedimensionality()
{
	return m_outputfeaturedimensionality;
}

void CDimensionChangingPreprocessor::set_inputfeaturedimensionality(const int32_t dim)
{
	m_inputfeaturedimensionality=dim;

	REQUIRE((m_inputfeaturedimensionality<=0)||(m_inputfeaturedimensionality!=m_outputfeaturedimensionality),"m_inputfeaturedimensionality should be not the same as m_outputfeaturedimensionality (or set to an invalid value: <=0)");
}

void CDimensionChangingPreprocessor::set_outputfeaturedimensionality(const int32_t dim)
{
	m_outputfeaturedimensionality=dim;

	REQUIRE((m_inputfeaturedimensionality<=0)||(m_inputfeaturedimensionality!=m_outputfeaturedimensionality),"m_inputfeaturedimensionality should be not the same as m_outputfeaturedimensionality (or set to an invalid value: <=0)");
}


template<class T>
int32_t CDimensionChangingPreprocessor::dimensioncheck(SGVector<T> sgvec1)
{
  int32_t len1=sgvec1.vlen;
  if(len1==get_inputfeaturedimensionality())
  {
	return 1;		
  }
  if(len1==get_outputfeaturedimensionality())
  {
	return 2;		
  }

	return 0;
}

int32_t CDimensionChangingPreprocessor::dimensioncheck2(const int32_t len1)
{
  if(len1==get_inputfeaturedimensionality())
  {
	return 1;		
  }
  if(len1==get_outputfeaturedimensionality())
  {
	return 2;		
  }

	return 0;
}
