#pragma once

#include <string>

namespace mrcv
{
	///////////////////////////////////////////////////////////////////////////////
	// Константы модуля UTILITY
	///////////////////////////////////////////////////////////////////////////////
	const bool IS_DEBUG_LOG_ENABLED = true;
	
	const std::string UTILITY_DEFAULT_RECORDER_FILENAME = "video";
	const int UTILITY_DEFAULT_RECORDER_INTERVAL = 5;
	const int UTILITY_DEFAULT_CAMERA_FPS = 25;
	///////////////////////////////////////////////////////////////////////////////
	// Структуры
	///////////////////////////////////////////////////////////////////////////////
	// Виды кодеков
	enum class CODEC
	{
		XVID,
		MJPG,
		mp4v,
		h265
	};
	// Виды записей в лог-файле
	enum class LOGTYPE
	{
		DEBUG,		// Отладака			DEBG
		ERROR,		// Ошибка			ERRR
		EXCEPTION,	// Исключение		EXCP
		INFO,		// Информация		INFO
		WARNING		// Предупреждение	WARN
	};
}
