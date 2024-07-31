#pragma once

#include <string>

namespace mrcv
{
	///////////////////////////////////////////////////////////////////////////////
	// ��������� ������ UTILITY
	///////////////////////////////////////////////////////////////////////////////
	const bool IS_DEBUG_LOG_ENABLED = true;
	
	const std::string UTILITY_DEFAULT_RECORDER_FILENAME = "video";
	const int UTILITY_DEFAULT_RECORDER_INTERVAL = 5;
	const int UTILITY_DEFAULT_CAMERA_FPS = 25;
	///////////////////////////////////////////////////////////////////////////////
	// ���������
	///////////////////////////////////////////////////////////////////////////////
	// ���� �������
	enum class CODEC
	{
		XVID,
		MJPG,
		mp4v,
		h265
	};
	// ���� ������� � ���-�����
	enum class LOGTYPE
	{
		DEBUG,		// ��������			DEBG
		ERROR,		// ������			ERRR
		EXCEPTION,	// ����������		EXCP
		INFO,		// ����������		INFO
		WARNING		// ��������������	WARN
	};
}
