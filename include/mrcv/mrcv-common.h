#pragma once

#include <string>

namespace mrcv
{
	// Константы модуля UTILITY
	const std::string UTILITY_DEFAULT_RECORDER_FILENAME = "video";
	const int UTILITY_DEFAULT_RECORDER_INTERVAL = 5;
	const int UTILITY_DEFAULT_CAMERA_FPS = 25;

	// Структуры
	enum class CODEC
	{
		XVID,
		MJPG,
		mp4v,
		h265
	};
}
